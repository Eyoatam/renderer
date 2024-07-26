#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const float PI = 3.141592654;
const int WIDTH = 4000;
const int HEIGHT = 4000;

typedef struct {
    int R;
    int G;
    int B;
    int A;
} Color;

typedef struct {
    unsigned char *data;
    int width;
    int height;
} Image;

typedef struct {
    float x;
    float y;
    float z;
} vec3f;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} vec4f;

typedef struct {
    int x;
    int y;
    int z;
} vec3i;

typedef struct {
    float x;
    float y;
    float z;
    Color color;
    vec3f normal;
} Vertex;

void identity(float *mat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void zero(float *mat) {
    for (int i = 0; i < 16; i++) {
        mat[i] = 0.0f;
    }
}

void copy_matrix(float *dest, float *src) {
    for (int i = 0; i < 16; i++) {
        dest[i] = src[i];
    }
}

void matmul(float *out, float *mat1, float *mat2) {
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float value = 0.0f;
            for (int k = 0; k < 4; k++) {
                value += mat1[i * 4 + k] * mat2[k * 4 + j];
            }
            out[i * 4 + j] = value;
        }
    }
}

void print_matrix(float *mat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", mat[i * 4 + j]);
        }
        printf("\n");
    }
}

void scale(float *mat, float factor) {
    float scale_matrix[16];
    identity(scale_matrix);
    scale_matrix[0] = factor;
    scale_matrix[5] = factor;
    scale_matrix[10] = factor;
    float out[16];
    zero(out);
    matmul(out, scale_matrix, mat);
    copy_matrix(mat, out);
}

void translate(float *mat, vec3f vec) {
    float translation_matrix[16];
    identity(translation_matrix);
    translation_matrix[3] = vec.x;
    translation_matrix[7] = vec.y;
    translation_matrix[11] = vec.z;

    float out[16];
    zero(out);
    matmul(out, translation_matrix, mat);
    copy_matrix(mat, out);
}

float radian(float degree) { return degree * (PI / 180.0f); }

void rotate(float *mat, float angle, vec3f axis) {
    float rotation_matrix[16];
    float c = cosf(angle);
    float s = sinf(angle);
    float i = 1 - c;

    // row major ordering
    rotation_matrix[0] = c + axis.x * axis.x * i;
    rotation_matrix[1] = axis.x * axis.y * i - axis.z * s;
    rotation_matrix[2] = axis.x * axis.z * i + axis.y * s;
    rotation_matrix[3] = 0.0f;
    rotation_matrix[4] = axis.y * axis.x * i + axis.z * s;
    rotation_matrix[5] = c + axis.y * axis.y * i;
    rotation_matrix[6] = axis.y * axis.z * i - axis.x * s;
    rotation_matrix[7] = 0.0f;
    rotation_matrix[8] = axis.z * axis.x * i - axis.y * s;
    rotation_matrix[9] = axis.z * axis.y * i + axis.x * s;
    rotation_matrix[10] = c + axis.z * axis.z * i;
    rotation_matrix[11] = 0.0f;
    rotation_matrix[12] = 0.0f;
    rotation_matrix[13] = 0.0f;
    rotation_matrix[14] = 0.0f;
    rotation_matrix[15] = 1.0f;
    float out[16];
    zero(out);
    matmul(out, rotation_matrix, mat);
    copy_matrix(mat, out);
}

void perspective(float *mat, float fov, float aspect_ratio, float near, float far) {
    float tangent = tan(fov / 2);
    zero(mat);
    mat[0] = 1.0f / (aspect_ratio * tangent);
    mat[5] = 1.0f / tangent;
    mat[10] = -(far + near) / (far - near);
    mat[11] = -(2.0f * far * near) / (far - near);
    mat[14] = -1.0f;
}

Color getpixel(unsigned char *image, int width, int x, int y) {
    unsigned int offset = (x + y * width) * 4;
    unsigned char r = image[offset];
    unsigned char g = image[offset + 1];
    unsigned char b = image[offset + 2];
    unsigned char a = image[offset + 3];
    return (Color){r, g, b, a};
}

void setpixel(unsigned char *image, int width, int x, int y, Color color) {
    unsigned int offset = (x + y * width) * 4;
    image[offset] = color.R;
    image[offset + 1] = color.G;
    image[offset + 2] = color.B;
    image[offset + 3] = color.A;
}

Color blend(int intensity, Color bg_color, Color line_color) {
    return (Color){((bg_color.R * (255 - intensity)) + intensity * line_color.R) / 255,
                   ((bg_color.G * (255 - intensity)) + intensity * line_color.G) / 255,
                   ((bg_color.B * (255 - intensity)) + intensity * line_color.B) / 255,
                   line_color.A};
}

// anti aliased line
// https://zingl.github.io/Bresenham.pdf
void draw_line(Vertex v0, Vertex v1, Image *image, Color color) {
    int x0 = v0.x;
    int y0 = v0.y;
    int x1 = v1.x;
    int y1 = v1.y;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int ed = dx + dy == 0 ? 1 : sqrt((float)dx * dx + (float)dy * dy);

    for (;;) {
        Color bg_color = getpixel(image->data, image->width, x0, y0);
        int intensity = 255 - (255 * abs(err - dx + dy) / ed);
        setpixel(image->data, image->width, x0, y0, blend(intensity, bg_color, color));
        if (x0 == x1 && y0 == y1) break;
        int e2 = err;
        int x2 = x0;
        if (2 * e2 > -dx) {
            intensity = 255 - (255 * (e2 + dy) / ed);
            bg_color = getpixel(image->data, image->width, x0, y0 + sy);
            if (e2 + dy < ed) {
                setpixel(image->data, image->width, x0, y0 + sy, blend(intensity, bg_color, color));
            }
            err -= dy;
            x0 += sx;
        }
        if (2 * e2 < dy) {
            intensity = 255 - (255 * (dx - e2) / ed);
            bg_color = getpixel(image->data, image->width, x2 + sx, y0);
            if (dx - e2 < ed) {
                setpixel(image->data, image->width, x2 + sx, y0, blend(intensity, bg_color, color));
            }
            err += dx;
            y0 += sy;
        }
    }
}

int min(int a, int b) { return (a < b) ? a : b; }
int max(int a, int b) { return (a > b) ? a : b; }
float minf(float a, float b) { return (a < b) ? a : b; }
float maxf(float a, float b) { return (a > b) ? a : b; }

vec3f cross(vec3f v1, vec3f v2) {
    float i = (v1.y * v2.z) - (v2.y * v1.z);
    float j = (v2.x * v1.z) - (v1.x * v2.z);
    float k = (v1.x * v2.y) - (v2.x * v1.y);
    return (vec3f){i, j, k};
}

float dot(vec3f v1, vec3f v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

vec3f normalize(vec3f vec) {
    float magnitude = sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    return (vec3f){vec.x / magnitude, vec.y / magnitude, vec.z / magnitude};
}

// doubled signed areae
int signed_area(Vertex a, Vertex b, Vertex c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

void draw_triangle(Image *image, Vertex v1, Vertex v2, Vertex v3, Color color) {
    int area = signed_area((Vertex){v1.x, v1.y}, (Vertex){v2.x, v2.y}, (Vertex){v3.x, v3.y});
    if (area < 0) return;
    draw_line(v1, v2, image, color);
    draw_line(v2, v3, image, color);
    draw_line(v3, v1, image, color);
}

void draw_triangle_filled(Image *image, Vertex v1, Vertex v2, Vertex v3, float *zbuffer,
                          Color ambient_color, Color diffuse_color, Color specular_color) {
    int area = signed_area(v1, v2, v3);
    if (area < 0) return;
    // bounding box
    int min_x = min(min(v1.x, v2.x), v3.x);
    int min_y = min(min(v1.y, v2.y), v3.y);
    int max_x = max(max(v1.x, v2.x), v3.x);
    int max_y = max(max(v1.y, v2.y), v3.y);
    // clamp bounding box to viewport boundaries
    min_x = max(0, min_x);
    min_y = max(0, min_y);
    max_x = min(max_x, WIDTH - 1);
    max_y = min(max_y, HEIGHT - 1);

    #pragma omp parallel for
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            // barycentric coordinates
            float bc1 = signed_area(v2, v3, (Vertex){x, y}) / (float)area;
            float bc2 = signed_area(v3, v1, (Vertex){x, y}) / (float)area;
            float bc3 = signed_area(v1, v2, (Vertex){x, y}) / (float)area;

            // phong reflection model
            // https://en.wikipedia.org/wiki/Phong_reflection_model
            vec3f vertex_normal = {0.0f, 0.0f, 0.0f};
            vertex_normal.x = bc1 * v1.normal.x + bc2 * v2.normal.x + bc3 * v3.normal.x;
            vertex_normal.y = bc1 * v1.normal.y + bc2 * v2.normal.y + bc3 * v3.normal.y;
            vertex_normal.z = bc1 * v1.normal.z + bc2 * v2.normal.z + bc3 * v3.normal.z;
            vertex_normal = normalize(vertex_normal);

            vec3f vertex_position = {0.0f, 0.0f, 0.0f};
            vertex_position.x = bc1 * v1.x + bc2 * v2.x + bc3 * v3.x;
            vertex_position.y = bc1 * v1.y + bc2 * v2.y + bc3 * v3.y;
            vertex_position.z = bc1 * v1.z + bc2 * v2.z + bc3 * v3.z;
            vertex_position = normalize(vertex_position);

            vec3f light_pos = normalize((vec3f){0.0f, 0.0f, -1.0f});
            vec3f light_vec =
                normalize((vec3f){light_pos.x - vertex_position.x, light_pos.y - vertex_position.y,
                                  light_pos.z - vertex_position.z});
            vec3f view_vec =
                normalize((vec3f){vertex_position.x, vertex_position.y,
                                  vertex_position.z - 2.0f}); // depends on the camera origin
            float ka = 1.0f;
            float kd = 1.0f;
            float ks = 0.2f;
            float n = 80.0f;
            float cosine = maxf(dot(light_vec, vertex_normal), 0.0f);
            float specular = 0.0f;
            if (cosine > 0.0f) {
                float tmp_a = 2 * dot(light_vec, vertex_normal);
                vec3f R = {tmp_a * vertex_normal.x - light_vec.x, //
                           tmp_a * vertex_normal.y - light_vec.y,
                           tmp_a * vertex_normal.z - light_vec.z};
                float proj_rv = maxf(dot(R, view_vec), 0.0f);
                specular = powf(proj_rv, n);
            }
            Color phong_color = {minf(ka * ambient_color.R + kd * cosine * diffuse_color.R +
                                          ks * specular * specular_color.R,
                                      255.0f),
                                 minf(ka * ambient_color.G + kd * cosine * diffuse_color.G +
                                          ks * specular * specular_color.G,
                                      255.0f),
                                 minf(ka * ambient_color.B + kd * cosine * diffuse_color.B +
                                          ks * specular * specular_color.B,
                                      255.0f),
                                 255};
            // convert to grayscale
            float intensity = (phong_color.R + phong_color.G + phong_color.B) / 765.0f;
            // fragment interpolation
            Color frag_color;
            frag_color.R = (bc1 * v1.color.R + bc2 * v2.color.R + bc3 * v3.color.R) * intensity;
            frag_color.G = (bc1 * v1.color.G + bc2 * v2.color.G + bc3 * v3.color.G) * intensity;
            frag_color.B = (bc1 * v1.color.B + bc2 * v2.color.B + bc3 * v3.color.B) * intensity;
            frag_color.A = 255;

            int is_inside =
                (bc1 >= 0 && bc1 <= 1) && (bc2 >= 0 && bc2 <= 1) && (bc3 >= 0 && bc3 <= 1);
            if (is_inside) {
                float z = v1.z * bc1 + v2.z * bc2 + v3.z * bc3;
                if (z < zbuffer[y * WIDTH + x]) {
                    zbuffer[y * WIDTH + x] = z;
                    setpixel(image->data, image->width, x, y, phong_color);
                }
            }
        }
    }
}

typedef struct {
    Vertex *vertices;
    vec3i *faces;
    int num_vertices;
    int num_faces;
} obj;

// a naive .obj file parser
obj load_model(const char *filename) {
    FILE *fp = fopen(filename, "r");
    assert(fp != NULL);
    char line[256];
    Vertex *vertices = (Vertex *)malloc(sizeof(vec3f));
    assert(vertices != NULL);
    vec3i *faces = (vec3i *)malloc(sizeof(vec3i));
    assert(faces != NULL);
    int vert_size = 1;
    int face_size = 1;

    while (fgets(line, sizeof(line), fp)) {
        float vx, vy, vz;
        switch (line[0]) {
            char trash[2];
        case 'v': {
            if (line[1] != ' ') break; // only consider geometric vertices
            sscanf(line, "%s %f %f %f", trash, &vx, &vy, &vz);
            vertices = (Vertex *)realloc(vertices, sizeof(Vertex) * (vert_size + 1));
            vertices[vert_size - 1] = (Vertex){vx, vy, vz == 0.0f ? vz + 1e-6f : vz};
            vertices[vert_size - 1].normal = (vec3f){0.0f, 0.0f, 0.0f};
            vert_size++;
            break;
        }
        case 'f': {
            int index1, index2, index3;
            sscanf(line, "%s %d %d %d", trash, &index1, &index2, &index3);
            faces = (vec3i *)realloc(faces, sizeof(vec3i) * (face_size + 1));
            faces[face_size - 1] = (vec3i){index1, index2, index3};
            face_size++;
            break;
        }
        }
    };
    obj model = (obj){vertices, faces, vert_size - 1, face_size - 1};
    fclose(fp);
    return model;
}

vec4f transform_to_clip(float *vertex, float *model, float *view, float *projection) {
    float trans[16], trans1[16], trans2[16];
    zero(trans);
    zero(trans1);
    zero(trans2);
    matmul(trans, model, vertex);
    matmul(trans1, view, trans);
    matmul(trans2, projection, trans1);
    return (vec4f){trans2[0], trans2[4], trans2[8], trans2[12]};
}

vec3f viewport_transform(vec4f vertex) {
    // perspective division
    vertex.x /= vertex.w;
    vertex.y /= vertex.w;
    vertex.z /= vertex.w;
    // viewport transform
    vertex.x = ((float)WIDTH / 2.0f) * vertex.x + (float)WIDTH / 2.0f;
    vertex.y = ((float)HEIGHT / 2.0f) * vertex.y + (float)HEIGHT / 2.0f;
    return (vec3f){vertex.x, vertex.y, vertex.z};
}

float *look_at(vec3f position, vec3f target, vec3f up) {
    vec3f direction = {position.x - target.x, position.y - target.y, position.z - target.z};
    vec3f right = normalize(cross(up, direction));
    float rot[16] = {
        right.x,     right.y,     right.z,     0.0f, //
        up.x,        up.y,        up.z,        0.0f, //
        direction.x, direction.y, direction.z, 0.0f, //
        0.0f,        0.0f,        0.0f,        1.0f  //
    };
    float translation[16];
    identity(translation);
    translate(translation, (vec3f){-position.x, -position.y, -position.z});
    float *out = (float *)malloc(16 * sizeof(float));
    assert(out != NULL);
    matmul(out, rot, translation);
    return out;
}

int main() {
    float model_mat[16];
    identity(model_mat);
    scale(model_mat, 0.0015f);
    rotate(model_mat, radian(-95.0f), (vec3f){1.0f, 0.0f, 0.0f});
    translate(model_mat, (vec3f){-1.0f, -0.25f, 0.0f});

    float *view =
        look_at((vec3f){0.0f, 0.0f, -2.0f}, (vec3f){0.0, 0.0, 0.0}, (vec3f){0.0f, 1.0f, 0.0f});
    float projection[16];
    perspective(projection, radian(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);

    obj model = load_model("obj/lucy.obj");
    unsigned char *data = (unsigned char *)malloc(WIDTH * HEIGHT * 4 * sizeof(char));
    assert(data != NULL);
    Image image = {data, WIDTH, HEIGHT};

    // Initialize zbuffer
    float *zbuffer = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        zbuffer[i] = FLT_MAX;
    }

    #pragma omp parallel for
    // calculate vertex normal
    // https://iquilezles.org/articles/normals/
    for (int i = 0; i < model.num_faces; i++) {
        vec3i face = model.faces[i];
        Vertex v1 = model.vertices[face.x - 1];
        Vertex v2 = model.vertices[face.y - 1];
        Vertex v3 = model.vertices[face.z - 1];

        float vert1[16] = {
            v1.x, 0.0f, 0.0f, 0.0f, //
            v1.y, 0.0f, 0.0f, 0.0f, //
            v1.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };
        vec4f v1_clip = transform_to_clip(vert1, model_mat, view, projection);

        float vert2[16] = {
            v2.x, 0.0f, 0.0f, 0.0f, //
            v2.y, 0.0f, 0.0f, 0.0f, //
            v2.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };
        vec4f v2_clip = transform_to_clip(vert2, model_mat, view, projection);

        float vert3[16] = {
            v3.x, 0.0f, 0.0f, 0.0f, //
            v3.y, 0.0f, 0.0f, 0.0f, //
            v3.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };
        vec4f v3_clip = transform_to_clip(vert3, model_mat, view, projection);

        vec3f normal =
            cross((vec3f){v3_clip.x - v1_clip.x, v3_clip.y - v1_clip.y, v3_clip.z - v1_clip.z},
                  (vec3f){v2_clip.x - v1_clip.x, v2_clip.y - v1_clip.y, v2_clip.z - v1_clip.z});

        model.vertices[face.x - 1].normal.x += normal.x;
        model.vertices[face.x - 1].normal.y += normal.y;
        model.vertices[face.x - 1].normal.z += normal.z;
        model.vertices[face.y - 1].normal.x += normal.x;
        model.vertices[face.y - 1].normal.y += normal.y;
        model.vertices[face.y - 1].normal.z += normal.z;
        model.vertices[face.z - 1].normal.x += normal.x;
        model.vertices[face.z - 1].normal.y += normal.y;
        model.vertices[face.z - 1].normal.z += normal.z;
    }

    #pragma omp parallel for
    for (int i = 0; i < model.num_faces; i++) {
        vec3i face = model.faces[i];
        Vertex v1 = model.vertices[face.x - 1];
        Vertex v2 = model.vertices[face.y - 1];
        Vertex v3 = model.vertices[face.z - 1];

        float vert1[16] = {
            v1.x, 0.0f, 0.0f, 0.0f, //
            v1.y, 0.0f, 0.0f, 0.0f, //
            v1.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };

        vec4f vert1_clip = transform_to_clip(vert1, model_mat, view, projection);
        vec3f vert1_screen = viewport_transform(vert1_clip);

        float vert2[16] = {
            v2.x, 0.0f, 0.0f, 0.0f, //
            v2.y, 0.0f, 0.0f, 0.0f, //
            v2.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };
        vec4f vert2_clip = transform_to_clip(vert2, model_mat, view, projection);
        vec3f vert2_screen = viewport_transform(vert2_clip);

        float vert3[16] = {
            v3.x, 0.0f, 0.0f, 0.0f, //
            v3.y, 0.0f, 0.0f, 0.0f, //
            v3.z, 0.0f, 0.0f, 0.0f, //
            1.0f, 0.0f, 0.0f, 0.0f  //
        };
        vec4f vert3_clip = transform_to_clip(vert3, model_mat, view, projection);
        vec3f vert3_screen = viewport_transform(vert3_clip);

        Color color = {255, 255, 255, 255};
        Vertex vert_1 = {(int)vert1_screen.x, (int)vert1_screen.y, vert1_screen.z, color,
                         v1.normal};
        Vertex vert_2 = {(int)vert2_screen.x, (int)vert2_screen.y, vert2_screen.z, color,
                         v2.normal};
        Vertex vert_3 = {(int)vert3_screen.x, (int)vert3_screen.y, vert3_screen.z, color,
                         v3.normal};
        Color ambient_color = {80, 80, 80, 255};
        Color diffuse_color = {35, 35, 35, 255};
        Color specular_color = {255, 255, 255, 255};
        draw_triangle_filled(&image, vert_1, vert_2, vert_3, zbuffer, ambient_color, diffuse_color,
                             specular_color);
        // draw_triangle(&image, vert_1, vert_2, vert_3, color); // wireframe mode
    }
    free(view);
    free(model.faces);
    free(model.vertices);
    free(zbuffer);
    stbi_flip_vertically_on_write(1); 
    stbi_write_png("output.png", image.width, image.height, 4, (void *)image.data, 0);
    free(data);
    return 0;
}

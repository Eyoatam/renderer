#include <float.h>
#include <math.h>
#include <stdio.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const float PI = 3.141592654;
const int WIDTH = 1000;
const int HEIGHT = 1000;

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
    int x;
    int y;
} vec2;

typedef struct {
    float x;
    float y;
} vec2f;

typedef struct {
    float x;
    float y;
    float z;
} vec3f;

typedef struct {
    int x;
    int y;
    int z;
} vec3i;

typedef struct {
    int x;
    int y;
    float z; // for calculating zbuffer
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

void zero_vec(float *vec) {
    for (int i = 0; i < 4; i++) {
        vec[i] = 0.0f;
    }
}

void copy_matrix(float *dest, float *src) {
    for (int i = 0; i < 16; i++) {
        dest[i] = src[i];
    }
}

void transpose(float *mat) {
    float tmp_mat[16];
    copy_matrix(tmp_mat, mat); 
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) continue;
             mat[i * 4 + j] = tmp_mat[j * 4 + i];
        }
    }
}

void matmul(float *out, float *mat1, float *mat2) {
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

void mat_vec_mul(float *out, float *vec, float *mat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i] += vec[j] * mat[j * 4 + i];
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
void draw_line(vec2 v0, vec2 v1, Image *image, Color color) {
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

vec3f cross(vec3f v1, vec3f v2) {
    float i = (v1.y * v2.z) - (v2.y * v1.z);
    float j = (v2.x * v1.z) - (v1.x * v2.z);
    float k = (v1.x * v2.y) - (v2.x * v1.y);
    return (vec3f){i, j, k};
}

vec3f normalize(vec3f vec) {
    float magnitude = sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    return (vec3f){vec.x / magnitude, vec.y / magnitude, vec.z / magnitude};
}

float dot(vec3f v1, vec3f v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

// doubled signed areae
int signed_area(Vertex a, Vertex b, Vertex c) {
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

void draw_triangle(Image *image, vec2 v1, vec2 v2, vec2 v3, Color color) {
    int area = signed_area((Vertex){v1.x, v1.y}, (Vertex){v2.x, v2.y}, (Vertex){v3.x, v3.y});
    if (area < 0) return;
    draw_line(v1, v2, image, color);
    draw_line(v2, v3, image, color);
    draw_line(v3, v1, image, color);
}

void draw_triangle_filled(Image *image, Vertex v1, Vertex v2, Vertex v3, vec3f normal,
                          float *zbuffer) {
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

    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            // barycentric coordinates
            float bc1 = signed_area(v2, v3, (Vertex){x, y}) / (float)area;
            float bc2 = signed_area(v3, v1, (Vertex){x, y}) / (float)area;
            float bc3 = signed_area(v1, v2, (Vertex){x, y}) / (float)area;
            // fragment interpolation
            Color pixel_color;
            pixel_color.R =
                (bc1 * v1.color.R + bc2 * v2.color.R + bc3 * v3.color.R); //  * intensity;
            pixel_color.G =
                (bc1 * v1.color.G + bc2 * v2.color.G + bc3 * v3.color.G); //  * intensity;
            pixel_color.B =
                (bc1 * v1.color.B + bc2 * v2.color.B + bc3 * v3.color.B); //  * intensity;
            pixel_color.A = 255;

            int is_inside =
                (bc1 >= 0 && bc1 <= 1) && (bc2 >= 0 && bc2 <= 1) && (bc3 >= 0 && bc3 <= 1);
            if (is_inside) {
                float z = v1.z * bc1 + v2.z * bc2 + v3.z * bc3;
                if (z < zbuffer[y * WIDTH + x]) {
                    zbuffer[y * WIDTH + x] = z;
                    setpixel(image->data, image->width, x, y, pixel_color);
                }
            }
        }
    }
}

typedef struct {
    vec3f *vertices;
    vec3i *faces;
    int num_vertices;
    int num_faces;
} obj;

// a naive .obj file parser
obj load_model(const char *filename) {
    FILE *fp = fopen(filename, "r");
    assert(fp != NULL);
    char line[256];
    vec3f *vertices = (vec3f *)malloc(sizeof(vec3f));
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
            vertices = (vec3f *)realloc(vertices, sizeof(vec3f) * (vert_size + 1));
            vertices[vert_size - 1] = (vec3f){vx, vy, vz == 0.0f ? vz + 1e-6f : vz};
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
    vec3f *tmp_v = vertices;
    obj model = (obj){tmp_v, faces, vert_size - 1, face_size - 1};
    fclose(fp);
    free(vertices);
    return model;
}

void viewport_transform(float *mat) {
    mat[0] /= mat[12];
    mat[4] /= mat[12];
    mat[8] /= mat[12];
    mat[0] = ((float)WIDTH / 2.0f) * mat[0] + (float)WIDTH / 2;
    mat[4] = ((float)HEIGHT / 2.0f) * mat[4] + (float)HEIGHT / 2;
}

int main() {
    float model_mat[16];
    identity(model_mat);
    // scale(model_mat, 0.5f);
    // scale(model_mat, 4.0f);
    // translate(model_mat, (vec3f){0.25f, -1.0f, 0.0f});
    // translate(model_mat, (vec3f){-0.05f, -0.20f, 0.0f});
    // translate(model_mat, (vec3f){0.0f, -0.50f, -6.0f});
    float view[16];
    identity(view);
    // translate(view, (vec3f){0.15f, -0.40f, 0.0f});
    // rotate(view, radian(-30.0f), (vec3f){0.0f, 1.0f, 0.0f});
    // rotate(view, radian(20.0f), (vec3f){0.0f, 1.0f, 0.0f});
    // rotate(view, radian(-10.0f), (vec3f){1.0f, 0.0f, 0.0f});
    // rotate(view, radian(-180.0f), (vec3f){0.0f, 1.0f, 0.0f});
    float projection[16];
    // identity(projection);
    //
    //
    // TODO: FIX PERSPECTIVE PROJECTION
    //
    //
    // perspective(projection, radian(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);

    printf("Before transpose\n");
    print_matrix(model_mat);
    transpose(model_mat);
    printf("After transpose\n");
    print_matrix(model_mat);
    transpose(view);
    transpose(projection);

    obj model = load_model("obj/teapot.obj");
    unsigned char *data = (unsigned char *)calloc(WIDTH * HEIGHT * 4, sizeof(char));
    Image image = (Image){data, WIDTH, HEIGHT};

    // set background
    for (int x = 0; x < image.width; x++) {
        for (int y = 0; y < image.height; y++) {
            setpixel(image.data, image.width, x, y, (Color){0, 0, 0, 255});
        }
    }

    float zbuffer[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        zbuffer[i] = FLT_MAX;
    }

    for (int i = 0; i < model.num_faces; i++) {
        vec3i face = model.faces[i];
        vec3f v1 = model.vertices[face.x - 1];
        vec3f v2 = model.vertices[face.y - 1];
        vec3f v3 = model.vertices[face.z - 1];

        float trans[4], trans1[4], trans2[4];
        zero_vec(trans);
        zero_vec(trans1);
        zero_vec(trans2);

        // first vertex
        float vert[4] = {v1.x, v1.y, v1.z, 1.0f};
        mat_vec_mul(trans, vert, model_mat);
        mat_vec_mul(trans1, trans, view);
        mat_vec_mul(trans2, trans1, projection);
        float v1x_b = trans2[0];
        float v1y_b = trans2[1];
        float v1z_b = trans2[2];
        // perspective division
        // trans2[0] /= trans2[3];
        // trans2[1] /= trans2[3];
        // trans2[2] /= trans2[3];
        // viewport transform
        trans2[0] = ((float)WIDTH / 2.0f) * trans2[0] + (float)WIDTH / 2;
        trans2[2] = ((float)HEIGHT / 2.0f) * trans2[2] + (float)HEIGHT / 2;
        float v1x = trans2[0];
        float v1y = trans2[1];
        float v1z = trans2[2];

        // second vertex
        zero_vec(trans);
        zero_vec(trans1);
        zero_vec(trans2);
        float vert2[4] = {v2.x, v2.y, v2.z, 1.0f};
        mat_vec_mul(trans, vert2, model_mat);
        mat_vec_mul(trans1, trans, view);
        mat_vec_mul(trans2, trans1, projection);
        float v2x_b = trans2[0];
        float v2y_b = trans2[1];
        float v2z_b = trans2[2];
        // perspective division
        // trans2[0] /= trans2[3];
        // trans2[1] /= trans2[3];
        // trans2[2] /= trans2[3];
        // viewport transform
        trans2[0] = ((float)WIDTH / 2.0f) * trans2[0] + (float)WIDTH / 2.0f;
        trans2[2] = ((float)HEIGHT / 2.0f) * trans2[2] + (float)HEIGHT / 2.0f;
        float v2x = trans2[0];
        float v2y = trans2[2];
        float v2z = trans2[3];
        // third vertex
        zero_vec(trans);
        zero_vec(trans1);
        zero_vec(trans2);
        float vert3[4] = {v3.x, v3.y, v3.z, 1.0f};
        mat_vec_mul(trans, vert3, model_mat);
        mat_vec_mul(trans1, trans, view);
        mat_vec_mul(trans2, trans1, projection);
        float v3x_b = trans2[0];
        float v3y_b = trans2[1];
        float v3z_b = trans2[2];
        // perspective division
        // trans2[0] /= trans2[3];
        // trans2[1] /= trans2[3];
        // trans2[2] /= trans2[3];
        // viewport transform
        trans2[0] = ((float)WIDTH / 2.0f) * trans2[0] + (float)WIDTH / 2.0f;
        trans2[2] = ((float)HEIGHT / 2.0f) * trans2[2] + (float)HEIGHT / 2.0f;
        float v3x = trans2[0];
        float v3y = trans2[2];
        float v3z = trans2[3];

        // surface normal
        vec3f normal = cross((vec3f){v3x_b - v1x_b, v3y_b - v1y_b, v3z_b - v1z_b},
                             (vec3f){v2x_b - v1x_b, v2y_b - v1y_b, v2z_b - v1z_b});
        normal = normalize(normal);
        float intensity = dot(normal, (vec3f){0.0f, 0.0f, -1.0f});
        Color color = {255 * intensity, 155 * intensity, 55 * intensity, 255};

        if (intensity > 0.0f) {
            Vertex vert_1 = (Vertex){(int)v1x, (int)v1y, v1z, color};
            Vertex vert_2 = (Vertex){(int)v2x, (int)v2y, v2z, color};
            Vertex vert_3 = (Vertex){(int)v3x, (int)v3y, v3z, color};
            draw_triangle_filled(&image, vert_1, vert_2, vert_3, normal, zbuffer);
        }
        // wireframe rendering
        // vec2 ver1 = (vec2){(int)v1x, (int)v1y};
        // vec2 ver2 = (vec2){(int)v2x, (int)v2y};
        // vec2 ver3 = (vec2){(int)v3x, (int)v3y};
        // draw_triangle(&image, ver1, ver2, ver3, (Color){255, 255, 255, 255});
    }

    stbi_flip_vertically_on_write(1); // set the origin to the bottom left corner
    stbi_write_png("output.png", image.width, image.height, 4, (void *)image.data, 0);
    free(data);
    return 0;
}

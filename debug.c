            // phong reflection model
            // vec3f vertex_normal = {0.0f, 0.0f, 0.0f};
            // vertex_normal.x = bc1 * normal.x + bc2 * normal.x + bc3 * normal.x;
            // vertex_normal.y = bc1 * normal.y + bc2 * normal.y + bc3 * normal.y;
            // vertex_normal.z = bc1 * normal.z + bc2 * normal.z + bc3 * normal.z;
            // // normalization
            // float norm_factor =
            //     sqrtf(vertex_normal.x * vertex_normal.x + vertex_normal.y * vertex_normal.y +
            //           vertex_normal.z * vertex_normal.z);
            // vertex_normal.x /= norm_factor;
            // vertex_normal.y /= norm_factor;
            // vertex_normal.z /= norm_factor;

            // vec3f light_vec = {0.0f, 0.0f, -1.0f};
            // vec3f view_vec = {0.0f, 0.0f, 1.0f};
            // float ka = 0.2f;
            // float kd = 0.5f;
            // float ks = 0.9f;
            // float n = 100.0f;
            // float ambient = ka * 0.1f;
            // float diffuse = dot(light_vec, vertex_normal) * kd * 1.0f;
            // float ln = 2 * dot(light_vec, vertex_normal);
            // vec3f reflection = {ln * vertex_normal.x - light_vec.x,
            //                     ln * vertex_normal.y - light_vec.y,
            //                     ln * vertex_normal.z - light_vec.z};
            // float specular = ks * powf(dot(reflection, view_vec), n) * 1.0f;
            // float intensity = ambient + diffuse + specular;


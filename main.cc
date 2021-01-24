#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "color.hh"
#include "ray.hh"
#include "scene.hh"
#include "theora.hh"
#include "vector.hh"
#include "random.hh"

using uniform_distribution = std::uniform_real_distribution<float>;
using color = Color<float>;
using object_ptr = std::unique_ptr<Object>;
using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

vec trace(ray r, const Object_group& objects) {
    float factor = 1;
    const int max_depth = 50;
    int depth=0;
    for (; depth<max_depth; ++depth) {
        if (Hit hit = objects.hit(r, 1e-3f, std::numeric_limits<float>::max())) {
            r = ray(hit.point, hit.normal + random_in_unit_sphere()); // scatter
            factor *= 0.5f; // diffuse 50% of light, scatter the remaining
        } else {
            break;
        }
    }
    //if (depth == max_depth) { return vec{}; }
    // nothing was hit
    // represent sky as linear gradient in Y dimension
    float t = 0.5f*(unit(r.direction())(1) + 1.0f);
    return factor*((1.0f-t)*vec(1.0f, 1.0f, 1.0f) + t*vec(0.5f, 0.7f, 1.0f));
}

void print_column_names(const char* version) {
    std::cout << std::setw(20) << "Time step";
    std::cout << std::setw(20) << "No. of steps";
    std::cout << std::setw(20) << version << " time";
    std::cout << '\n';
}

void ray_tracing_cpu() {
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    int nx = 2100, ny = 1400, nrays = 100;
    Pixel_matrix<float> pixels(nx,ny);
    thx::screen_recorder recorder("out.ogv", nx,ny);
    Object_group objects;
    objects.add(object_ptr(new Sphere(vec(0.f,0.f,-1.f),0.5f)));
    objects.add(object_ptr(new Sphere(vec(0.f,-1000.5f,-1.f),1000.f)));
    Camera camera;
    uniform_distribution distribution(0.f,1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenMP");
    duration total_time = duration::zero();
    for (int time_step=1; time_step<=max_time_step; ++time_step) {
        auto t0 = clock_type::now();
        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int j=0; j<ny; ++j) {
            for (int i=0; i<nx; ++i) {
                vec sum;
                for (int k=0; k<nrays; ++k) {
                    float u = (i + distribution(prng)) / nx;
                    float v = (j + distribution(prng)) / ny;
                    sum += trace(camera.make_ray(u,v),objects);
                }
                sum /= float(nrays); // antialiasing
                sum = pow(sum,1.f/gamma); // gamma correction
                pixels(i,j) = to_color(sum);
            }
        }
        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1-t0);
        total_time += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << max_time_step
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out.ppm");
        out << pixels;
        recorder.record_frame(pixels);
        camera.move(vec{0.f,0.f,0.1f});
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(total_time).count()
        << "s" << std::endl;
    std::clog << "Movie time: " << max_time_step/60.f << "s" << std::endl;
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

struct opencl_vector {
    float x;
    float y;
    float z;
    float w;
};

struct opencl3_vector {
    float x;
    float y;
    float z;
};


void ray_tracing_gpu(OpenCL& opencl){

    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    int nx = 2100, ny = 1400, nrays = 100;
    Pixel_matrix<float> pixels(nx,ny);
    thx::screen_recorder recorder("out.ogv", nx,ny);
    std::vector<Sphere> objects = {
        Sphere{vec(0.f,0.f,-1.f),0.5f},
        Sphere{vec(0.f,-1000.5f,-1.f),1000.f}
    };
    Camera camera;
    uniform_distribution distribution(0.f,1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenCL");

    std::vector<float> un_real_random;
    std::vector<float> uniform_random;
    int random_length = 1 << 25;


    std::uniform_real_distribution<float> un_real_distribution(-1.0,1.0);


    for (int i =0; i < random_length ;++i){
        un_real_random.push_back(un_real_distribution(prng));
        uniform_random.push_back(distribution(prng));
    }

    duration total_time = duration::zero();
    
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, nx*ny*3*sizeof(float));
    cl::Buffer d_rand_un_real(opencl.context, begin(un_real_random), end(un_real_random), true);
    cl::Buffer d_rand_uniform(opencl.context, begin(uniform_random), end(uniform_random), true);
    cl::Kernel kernel(opencl.program, "perform_ray_tracing");
    

    kernel.setArg(0, d_result);
    kernel.setArg(1, nx);
    kernel.setArg(2, ny);

    kernel.setArg(3, opencl_vector{objects[0].origin()(0), objects[0].origin()(1), objects[0].origin()(2), objects[0].radius()});

    kernel.setArg(4, opencl_vector{objects[1].origin()(0), objects[1].origin()(1), objects[1].origin()(2), objects[1].radius()});

    kernel.setArg(5, d_rand_un_real);
    kernel.setArg(6, d_rand_uniform);
    kernel.setArg(7, nrays);


    for (int time_step=1; time_step<=max_time_step; ++time_step) {
        kernel.setArg(8, opencl_vector{ camera.origin(0), camera.origin(1), camera.origin(2), 0});

        auto t0 = clock_type::now();

        // TODO Use GPU to race sun rays!

        opencl.queue.flush();
        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nx,ny), cl::NullRange);
        opencl.queue.flush();
        opencl.queue.enqueueReadBuffer(d_result, true, 0, 3*nx*ny*sizeof(float), (float*)(pixels.pixels().data()));        

        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1-t0);
        total_time += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << max_time_step
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out.ppm");
        out << pixels;
        recorder.record_frame(pixels);
        camera.move(vec{0.f,0.f,0.1f});
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(total_time).count()
        << "s" << std::endl;
    std::clog << "Movie time: " << max_time_step/60.f << "s" << std::endl;


}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    ray_tracing_gpu(opencl);

}


const std::string src = R"(
typedef struct {
    float3 origin;
    float3 direction;
} Ray;

typedef struct {
    float t;
    float3 point;
    float3 normal;
} Hit;

bool hit(Hit* hit) { //check if hit is true
    return hit->t > 0;
}

Ray create_ray(float3 _origin, float3 _direction) {
    Ray ray;
    ray.origin = _origin;
    ray.direction = _direction;
    return ray;
}

Ray make_ray(float u, float v, float3 origin) {
    float3 _lower_left_corner = (float3)(3.02374f,-1.22628f,3.4122f);
    float3 _horizontal = (float3)(1.18946f,0.f,-5.15434f);
    float3 _vertical = (float3)(-0.509421f,3.48757f,-0.117559f);
    float3 dir = _lower_left_corner + (u * _horizontal) + (v * _vertical) - origin;
    return create_ray(origin, dir);
}

Hit hit_sphere(Ray* ray, float t_min, float t_max, float4 sphere) {

    Hit result;
    result.t = -1.f;
    float3 oc = ray->origin - sphere.xyz;
    float a = dot(ray->direction, ray->direction);
    float b = dot(oc, ray->direction);
    float c = dot(oc, oc) - sphere.w * sphere.w;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float d = sqrt(discriminant); 
        float t = (-b - d)/a;
        bool success = false;
        if (t_min < t && t < t_max) {
            success = true;
        } else {
            t = (-b + d)/a;
            if (t_min < t && t < t_max) { success = true; }
        }
        if (success) {
            result.t = t;
            result.point = ray->origin + t*ray->direction;
            result.normal = (result.point - sphere.xyz) / sphere.w;
        }
    }
    return result;
}

Hit hit_both_spheres(Ray* ray, float min, float max, float4 big_sphere, float4 small_sphere) {
    Hit result; 
    result.t = -1.f;
 
    Hit s1 = hit_sphere(ray, min, max, small_sphere);

    if (hit(&s1)) {
        max = s1.t;
        result = s1;
    }

    Hit s2 = hit_sphere(ray, min, max, big_sphere);

    if (hit(&s2)) {
        max = s2.t;
        result = s2;
    }

    return result;
}


float3 random_in_unit_sphere(global float* un_random, int seed) {

    long rand_size = (1<<22) - 3;
    long rand_id = (get_global_id(1)*seed + get_global_id(0)) % rand_size;
    float4 result_rand;
    float3 result;
    float square_x= 3;
    //printf("%f \n", un_random[rand_id]);
    do {
        result_rand.x = un_random[rand_id], result_rand.y = un_random[2 * rand_id + seed], result_rand.z = un_random[3 * rand_id + seed], result_rand.w = un_random[4 * rand_id - seed];
        square_x = result_rand.x * result_rand.x + result_rand.y * result_rand.y + result_rand.z * result_rand.z + result_rand.w * result_rand.w;
        rand_id = (rand_id) %  rand_size - seed; 
        seed = (seed + 2) % 1000;
        //printf("%f \n", square_x);
    } while (square_x >= 1.f);
    // printf("End while in random_in_unit_sphere");

    result.x = 2 * (result_rand.y *  result_rand.w +  result_rand.x *  result_rand.z) / square_x;
    result.y = 2 * (result_rand.z *  result_rand.w -  result_rand.x *  result_rand.y) / square_x;
    result.z = (result_rand.x * result_rand.x + result_rand.w * result_rand.w - result_rand.y * result_rand.y - result_rand.z * result_rand.z) / square_x;
    return result;
}


float3 trace_ray(Ray* r, float4 big_sphere, float4 small_sphere, global float* un_random, int number_of_ray) {
    
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    
    float factor = 1;
    const int max_depth = 50;
    int depth=0;
    for (; depth < max_depth; ++depth) {
        Hit h = hit_both_spheres(r, 1e-3f, FLT_MAX, big_sphere, small_sphere);
        if (hit(&h)) {
            r->origin = h.point;
            r->direction = h.normal;
            int seed = (int) depth + un_random[0] * r->direction.x + number_of_ray;
            seed = seed % 1000;
                       
            float3 rand = random_in_unit_sphere(un_random, seed);
            
            if (global_id_x == 1 && global_id_y == 1){
                //printf("Direction of ray  %f %f %f \n", r->direction.x ,  r->direction.y,  r->direction.z  );
               // printf("Result of random_in_unit_sphere  %f %f %f \n",rand.x , rand.y, rand.z  );
             }
            r->direction += rand;

            factor *= 0.5f; // diffuse 50% of light, scatter the remaining
        } else {
            break;
        }
    }

    r->direction /= length(r->direction);
    float t = 0.5f*(r->direction.y + 1.0f);
    return factor*((1.0f-t)*(float3)(1.0f, 1.0f, 1.0f) + t*(float3)(0.5f, 0.7f, 1.0f));
}

kernel void perform_ray_tracing(global float* pixels, int nx, int ny, float4 small_sphere, float4 big_sphere, global float* un_real_random, global float* uniform_random, int nrays, float3 origin) {
        
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    int global_id = global_id_y*nx + global_id_x;
    //printf("Nrays received in OpenCl %i \n", nrays);
    float3 sum = 0;

    for (int i = 0; i < nrays; i++) {
        float u = (float)(global_id_x + uniform_random[2*(global_id+i)+0]) / nx; 
        float v = (float)(global_id_y + uniform_random[2*(global_id+i)+1]) / ny;
        
        Ray r = make_ray(u, v, origin);
        sum += trace_ray(&r, big_sphere, small_sphere, un_real_random, i);
        //printf("End 1 trace ray \n");

    }
    //printf("End for in perform_ray_tracing \n");
    sum /= (float)(nrays) + 0.001f;
    float gamma = 2.0f;
    sum = pow(sum, 1.f/gamma); 
    pixels[3 * (nx*global_id_y+global_id_x)+0] = sum.x;
    pixels[3 * (nx*global_id_y+global_id_x)+1] = sum.y;
    pixels[3 * (nx*global_id_y+global_id_x)+2] = sum.z;
}


)";


void ray_tracing_opencl() {
    //std::clog << "GPU version is not implemented!" << std::endl; std::exit(1);
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
           // return 1;
        }

        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        
    cl::CommandQueue queue(context, device);
    OpenCL opencl{platform, device, context, program, queue};
    opencl_main(opencl);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
       // return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
       // return 1;
    }    
}


int main(int argc, char* argv[]) {
    enum class Version { CPU, GPU };
    Version version = Version::CPU;
    if (argc == 2) {
        std::string str(argv[1]);
        for (auto& ch : str) { ch = std::tolower(ch); }
        if (str == "gpu") { version = Version::GPU; }
    }
    switch (version) {
        case Version::CPU: ray_tracing_cpu(); break;
        case Version::GPU: ray_tracing_opencl(); break;
        default: return 1;
    }
    return 0;
}

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
using Vec = Eigen::Vector3f;
using Mat = Eigen::Matrix3f;
using Vec4 = Eigen::Vector4f;
using Vec3i = Eigen::Vector3i;
using namespace std;


int framenum = 0;
const int n = 100;
const float dt = 1e-4f;
const float frame_dt = 1e-3f;
const float dx = 1.0f / n;
const float inv_dx = 1.0f / dx;

const auto particle_mass = 1.0f;
const auto vol = 1.0f;
const auto hardening = 5.0f;
const auto E = 2e3f;
const auto nu = 0.3f;
const bool plastic = true;

// Lame
const float mu_0 = E / (2 * (1 + nu));
const float lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

struct Particle
{
    Vec x, v;
    Mat F;
    Mat C;
    float Jp;
    int c;
    float E;
    bool plastic;
    Particle(Vec x,
             int c,
             float E = 1e3f,
             bool plastic = false,
             Vec v = Vec(0, 0, 0)) : x(x),
                                     v(v),
                                     F(Mat::Identity()),
                                     C(Mat::Zero()),
                                     Jp(1),
                                     c(c),
                                     E(E),
                                     plastic(plastic) {}
};

Eigen::Vector4f grid[n + 1][n + 1][n + 1];
std::vector<Particle> particles;

void add_object(Vec center, int c, Vec scale = Vec::Ones(), float E = 1e3f, bool plastic = false, int sample = 40000)
{
    for (int n = 0; n < sample; n++)
    {
        particles.push_back(Particle(Vec::Random().cwiseProduct(scale) * 0.1 + center, c, E, plastic));
    }
}

void sub_step(float dt)
{

    std::memset(grid, 0, sizeof(grid));

    //P2G
    for (auto &p : particles)
    {
        Eigen::Vector3i base = (p.x * inv_dx - Vec(0.5f, 0.5f, 0.5f)).cast<int>();
        Vec x = p.x * inv_dx - base.cast<float>();
        Vec w[3] = {
            Vec(0.5f, 0.5f, 0.5f).cwiseProduct((Vec)(Vec(1.5f, 1.5f, 1.5f) - x).array().square()),
            Vec(0.75f, 0.75, 0.75) - (Vec)(x - Vec(1.0, 1.0, 1.0)).array().square(),
            Vec(0.5f, 0.5, 0.5).cwiseProduct((Vec)(x - Vec(0.5, 0.5, 0.5)).array().square())};

        float mu = p.E / (2 * (1 + nu));
        float lambda = p.E * nu / ((1 + nu) * (1 - 2 * nu));

        auto e = std::exp(hardening * (1.0f - p.Jp));
        float J = p.F.determinant();

        //polar decomposition
        Mat r, s;
        Eigen::JacobiSVD<Mat> svd(p.F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat u, v, sig;
        u = svd.matrixU();
        v = svd.matrixV();
        sig = svd.singularValues().asDiagonal();
        r = u * v.transpose();
        v = v * sig * v.transpose();
        // polar_decomp(p.F, r, s);
        // std::cout << p.F;
        // std::cout << r;
        // std::cout << s;
        //fix corotated model
        Mat P = 2 * mu * (p.F - r) + lambda * (J - 1) * J * p.F.inverse().transpose();
        Mat PF = P * p.F.transpose();
        float Dinv = 4 * inv_dx * inv_dx;
        Mat stress = -(dt * vol) * (Dinv * PF);
        Mat affine = stress + particle_mass * p.C;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                {
                    Vec dpos = (Vec(i, j, k) - x) * dx;
                    Vec momentum = p.v * particle_mass;
                    Eigen::Vector4f mass_x_velocity(momentum[0], momentum[1], momentum[2], particle_mass);
                    Vec tmp = affine * dpos;
                    float weight = w[i].x() * w[j].y() * w[k].z();
                    grid[base.x() + i][base.y() + j][base.z() + k] += weight *
                                                                      (mass_x_velocity + Eigen::Vector4f(tmp[0], tmp[1], tmp[2], 0));
                }
    }

    //grid computation
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++)
            for (int k = 0; k <= n; k++)
            {
                auto &g = grid[i][j][k];
                if (g[3] > 0)
                {
                    g /= g[3];
                    g += dt * Vec4(0, -200, 0, 0);
                    float boundary = 0.05;
                    float x = float(i) / n;
                    float y = float(j) / n;
                    if (x < boundary || x > 1 - boundary || y > 1 - boundary)
                    {
                        g = Vec4(0, 0, 0, 0);
                    }
                    // Separate boundary
                    if (y < boundary)
                    {
                        g[1] = std::max(0.0f, g[1]);
                    }
                }
            }

    //G2P
    for (auto &p : particles)
    {
        p.v = Vec::Zero();
        p.C = Mat::Zero();
        Vec3i base = (p.x * inv_dx - Vec(0.5f, 0.5f, 0.5f)).cast<int>();
        Vec x = p.x * inv_dx - base.cast<float>();
        Vec w[3] = {
            Vec(0.5f, 0.5f, 0.5f).cwiseProduct((Vec)(Vec(1.5f, 1.5f, 1.5f) - x).array().square()),
            Vec(0.75f, 0.75, 0.75) - (Vec)(x - Vec(1.0, 1.0, 1.0)).array().square(),
            Vec(0.5f, 0.5, 0.5).cwiseProduct((Vec)(x - Vec(0.5, 0.5, 0.5)).array().square())};

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {

                    auto dpos = (Vec(i, j, k) - x) * dx;
                    auto grid_v = grid[base.x() + i][base.y() + j][base.z() + k].head<3>();
                    auto weight = w[i].x() * w[j].y() * w[k].z();
                    p.v += weight * grid_v;
                    p.C += 4 * inv_dx * inv_dx * weight * grid_v * dpos.transpose();
                }
            }
        }
        //damping
        auto sym = 0.5f * (p.C + p.C.transpose());
        auto skew = p.C - sym;
        float rdamping = 0.6;
        float adamping = 0.1;
        p.C = (1 - rdamping) * sym + (1 - adamping) * skew;

        //advection
        p.x += dt * p.v;
        Mat F = (Mat::Identity() + dt * p.C) * p.F;
        p.F = F;

        // Mat svd_u, sig, svd_v;
        // Eigen::JacobiSVD<Mat> svd(p.F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // svd_u = svd.matrixU();
        // svd_v = svd.matrixV();
        // sig = svd.singularValues().asDiagonal();

        // // Plasticity
        // for (int i = 0; i < 3 * int(p.plastic); i++)
        // {
        //     sig(i, i) = clamp(sig(i, i), 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
        // }

        // float oldJ = F.determinant();
        // F = svd_u * sig * svd_v.transpose();

        // float Jp_new = clamp(p.Jp * oldJ / F.determinant(), 0.6f, 20.0f);

        // p.Jp = Jp_new;
        // p.F = F;
    }
}

void saveobj()
{
    std::ofstream myfile;
    std::stringstream filename;
    filename << "output/" << framenum << ".obj";
    cout << filename.str() << endl;
    myfile.open(filename.str());
    for (auto &p : particles)
    {
        std::stringstream s;
        s << "v " << p.x.x() << " " << p.x.y() << " " << p.x.z();
        myfile << s.str() << std::endl;
    }
    myfile.close();
}

int main(int argc, char const *argv[])
{

    add_object(Vec(0.55, 0.5, 0.55), 0xED533B, Vec(1, 1, 1), 1e3, true);
    // add_object(Vec(0.55, 0.45, 0.55), 0xF2B134, Vec(0.5,0.5,0.5),1e4, false, 1000);
    cout << "start simulation" << endl;
    for (int step = 0;; step++)
    {
        sub_step(dt);
        if (step % int(frame_dt / dt) == 0)
        {
            cout << "step" << endl;
            saveobj();
            framenum++;
        }
    }
}

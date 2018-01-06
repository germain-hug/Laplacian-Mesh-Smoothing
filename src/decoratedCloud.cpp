//
// Created by bontius on 20/01/17.
//

#include "acq/impl/decoratedCloud.hpp"
#include "igl/invert_diag.h"

#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>

namespace acq {

    DecoratedCloud::DecoratedCloud(CloudT const &vertices)
            : _vertices(vertices) {}

    DecoratedCloud::DecoratedCloud(CloudT const &vertices, FacesT const &faces)
            : _vertices(vertices), _faces(faces) {}

    DecoratedCloud::DecoratedCloud(CloudT const &vertices, FacesT const &faces, NormalsT const &normals)
            : _vertices(vertices), _faces(faces), _normals(normals) {}

    DecoratedCloud::DecoratedCloud(CloudT const &vertices, NormalsT const &normals)
            : _vertices(vertices), _normals(normals) {}


    /**  ======================================== **/
    /**  ========== I) a) MEAN CURVATURE ======== */
    /**  ======================================== **/

    Eigen::VectorXd DecoratedCloud::meanCurvature() {

        /// ---- 1) Retrieve Vertices and compute Neighbors
        Eigen::MatrixXd V1 = this->getVertices();
        Eigen::MatrixXi F1 = this->getFaces();

        acq::NeighboursT neighbours = computeNeighbours(F1, V1);

        /// ---- 2) Compute Uniform Laplace Operator
        Eigen::SparseMatrix<double> L = computeUniformLaplace(F1, V1, neighbours);

        /// ---- 3) Compute Mean Curvature
        Eigen::VectorXd H = 0.5 * (L * V1).rowwise().norm();

        /// ---- 4) Orient Mean Curvature
        orientMeanCurvature(H, neighbours, V1);
        return H;
    }



    /**  ============================================ **/
    /**  ========== I) b) GAUSSIAN CURVATURE ======== */
    /**  ============================================ **/

    Eigen::VectorXd DecoratedCloud::gaussianCurvature() {

        /// ---- Retrieve Vertices ---- ///
        Eigen::MatrixXd V1 = this->getVertices();
        Eigen::MatrixXi F1 = this->getFaces();
        Eigen::VectorXd K(V1.rows());

        for (int i = 0; i < V1.rows(); i++) {
            /// ---- See computeArea_Angle for more details ---- ///
            std::tuple<double, double> a_and_a = computeArea_Angle(F1, V1, i);
            double A_i = std::get<0>(a_and_a);
            K(i) = std::get<1>(a_and_a) / (A_i / 3.0);
        }

        return K;
    }




    /**  ============================================ **/
    /**  ======== II) DISCRETE MEAN CURVATURE ======= **/
    /**  ============================================ **/

    Eigen::VectorXd DecoratedCloud::discreteMeanCurvature(
            Eigen::MatrixXi TT,
            Eigen::MatrixXi TTi,
            double lambda,
            bool exp_smoothing,
            bool imp_smoothing
    ) {

        /// Retrieve Vertices and Faces
        Eigen::MatrixXd V1 = this->getVertices();
        Eigen::MatrixXi F1 = this->getFaces();
        const int n = V1.rows();
        Eigen::SparseMatrix<double> L(n,n),C_mat(n,n),M_inv(n,n),M(n,n);

        for (int i = 0; i < V1.rows(); i++) {

            /// ---- Compute Area ----
            std::tuple<double, double> a_and_a = computeArea_Angle(F1, V1, i);
            double A_i = std::get<0>(a_and_a)/3;
            double w_ij_sum = 0;

            for (int j = 0; j < F1.rows(); j++) {

                int next = -1, prev = -1, edge = -1;
                if (F1.row(j)(0) == i) {
                    next = F1.row(j)(1);
                    prev = F1.row(j)(2);
                    edge = 2;
                } else if (F1.row(j)(1) == i) {
                    next = F1.row(j)(2);
                    prev = F1.row(j)(0);
                    edge = 0;
                } else if (F1.row(j)(2) == i) {
                    next = F1.row(j)(0);
                    prev = F1.row(j)(1);
                    edge = 1;
                }

                if (edge!=-1) { // If we have found a neighbour
                    Eigen::RowVector3d A = V1.row(i);
                    Eigen::RowVector3d B = V1.row(next);
                    Eigen::RowVector3d C = V1.row(prev);

                    /// ---- Compute Angle beta_ij ----
                    double beta = compute_theta(A,B,C);

                    /// ---- Find Adjacent Triangles ----
                    int adj_triangle_idx = TT(j, edge); // Retrieve Adjacent Triangle
                    int adj_edge_idx = TTi(j, edge);    // Retrieve Adj. edge index in Adj. Triangle

                    Eigen::RowVector3i F2 = F1.row(adj_triangle_idx);
                    if (adj_edge_idx == 0) {
                        A = V1.row(F2(0));
                        B = V1.row(F2(1));
                        C = V1.row(F2(2));
                    } else if (adj_edge_idx == 1) {
                        A = V1.row(F2(1));
                        B = V1.row(F2(2));
                        C = V1.row(F2(0));
                    } else if (adj_edge_idx == 2) {
                        A = V1.row(F2(2));
                        B = V1.row(F2(0));
                        C = V1.row(F2(1));
                    }

                    /// ---- Compute Angle alpha_ij ----
                    double alpha = compute_theta(A,C,B);

                    /// ---- Compute Cotan Weight w_ij ----
                   // double w_ij = cos(alpha) / sin(alpha) + cos(beta) / sin(beta);
                    double w_ij = tan(M_PI/2 - alpha) + tan(M_PI/2 - beta);
                    /// ---- Fill Laplace Matrix -----
                    C_mat.insert(i, prev) = w_ij;
                    w_ij_sum += w_ij;
                }
            }
            C_mat.insert(i, i) = -w_ij_sum;
            M.insert(i, i) = 2.0 * A_i;
            M_inv.insert(i, i) = 1.0/ (2.0 * A_i);
        }

        /// ---- Compute Laplace Operator ----
        L = M_inv * C_mat;

        /// ---- Explicit Laplacian Smoothing -----
        if (exp_smoothing) {
            Eigen::SparseMatrix<double> I(n,n);
            I.setIdentity();
            this->setVertices((I + lambda * L) * V1);
        }


        /// ---- Implicit Laplacian Smoothing -----
        if (imp_smoothing) {

            /// ---- Compute 'A' Matrix ----
            Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(M - lambda * M * L);

            /// ---- Compute 'b' Vectors and solve ----
            Eigen::VectorXd x1 = chol.solve(M * V1.col(0));
            Eigen::VectorXd x2 = chol.solve(M * V1.col(1));
            Eigen::VectorXd x3 = chol.solve(M * V1.col(2));
            Eigen::MatrixXd x(n, 3);
            x << x1, x2, x3;

            this->setVertices(x);
        }

        /// ---- Compute Mean Curvature
        Eigen::VectorXd H = 0.5 * (L * V1).rowwise().norm();

        /// ---- Orient Mean Curvature ----
        acq::NeighboursT neighbours = computeNeighbours(F1, V1);
        orientMeanCurvature(H, neighbours, V1);


        return H;
    }

    /**  ===================================== **/
    /**  ============= ADD NOISE ============= **/
    /**  ===================================== **/

    void DecoratedCloud::addNoise(double factor){

        // ----- Get Vertices ----
        Eigen::MatrixXd V2 = this->getVertices();

        // ----- Initial Randomizer ----
        std::srand(std::time(0));

        // Compute Noise Variance according to BB
        double n_x = V2.col(0).maxCoeff() - V2.col(0).minCoeff();
        double n_y = V2.col(1).maxCoeff() - V2.col(1).minCoeff();
        double n_z = V2.col(2).maxCoeff() - V2.col(2).minCoeff();


        for (int i = 0; i < int(V2.rows()); i++) {
            V2.row(i) = V2.row(i) + Eigen::RowVector3d(
                    factor * n_x * ((double) std::rand() / (RAND_MAX)),
                    factor * n_y * ((double) std::rand() / (RAND_MAX)),
                    factor * n_z * ((double) std::rand() / (RAND_MAX)));
        } 
        // ----- Update and Display Mesh
        this->setVertices(V2);
    }



    /**  ===================================== **/
    /**  ============= FUNCTIONS ============= **/
    /**  ===================================== **/


    /// ==============================================
    /// ====== -> Compute Angle btw Vertices <- ======
    /// ==============================================


    inline double DecoratedCloud::compute_theta(Eigen::RowVector3d A, Eigen::RowVector3d B,Eigen::RowVector3d C){
        return acos((A - B).dot(C - B) / ((A - B).norm() * (C - B).norm()));
    }

    /// ======================================
    /// ====== -> Compute Neighbours <- ======
    /// ======================================

    acq::NeighboursT DecoratedCloud::computeNeighbours(Eigen::MatrixXi &F1, Eigen::MatrixXd &V1) {
        acq::NeighboursT neighbours;
        for (int i = 0; i < V1.rows(); i++) {
            acq::NeighboursT::mapped_type currNeighbours;
            for (int j = 0; j < F1.rows(); j++) {
                /* Get Current Face */
                Eigen::RowVector3i curr_face = F1.row(j);
                if (curr_face(0) == i) {
                    currNeighbours.insert(curr_face(1));
                    currNeighbours.insert(curr_face(2));
                } else if (curr_face(1) == i) {
                    currNeighbours.insert(curr_face(0));
                    currNeighbours.insert(curr_face(2));
                } else if (curr_face(2) == i) {
                    currNeighbours.insert(curr_face(0));
                    currNeighbours.insert(curr_face(1));
                }
            }
            // Store list of neighbours
            std::pair<acq::NeighboursT::iterator, bool> const success =
                    neighbours.insert(std::make_pair(i, currNeighbours));
        }
        return neighbours;
    }



    /// ====================================================
    /// ====== -> Compute Uniform Laplace Operator <- ======
    /// ====================================================

    Eigen::SparseMatrix<double> DecoratedCloud::computeUniformLaplace(
            Eigen::MatrixXi &F1,
            Eigen::MatrixXd &V1,
            acq::NeighboursT &neighbours
    ) {
        const long n = V1.rows();
        Eigen::SparseMatrix<double> L(n, n);
        for (int i = 0; i < n; i++) {
            const std::set<size_t> neighbours_ID = neighbours.at(i);
            for (auto j : neighbours_ID) {
                L.insert(i, j) = 1.0 / double(neighbours_ID.size());
            }
            L.insert(i, i) = -1.0;
        }
        return L;
    }

    /// =========================================
    /// ====== -> Orient Mean Curvature <- ======
    /// =========================================

    void DecoratedCloud::orientMeanCurvature(
            Eigen::VectorXd &H,
            acq::NeighboursT &neighbours,
            Eigen::MatrixXd &V1
    ) {

        Eigen::MatrixXd N = this->_normals;

        for (int i = 0; i < V1.rows(); i++) {

            const std::set<size_t> neighbours_ID = neighbours.at(i);
            Eigen::RowVector3d average;
            average.setZero();

            for (auto j : neighbours_ID) {
                Eigen::RowVector3d current_row = V1.row(j);
                average += current_row / neighbours_ID.size();
            }
            // If Normal and Mean Curvature are in opposite directions, invert sign
            if (N.row(i).dot(average - V1.row(i)) > 0) {
                H.row(i) = -H.row(i);
            }
        }
    }

    /// ==================================================
    /// ====== -> Compute Area and Angle Deficit <- ======
    /// ==================================================

    std::tuple<double, double> DecoratedCloud::computeArea_Angle(
            Eigen::MatrixXi &F1,
            Eigen::MatrixXd &V1,
            int i
    ) {

        Eigen::RowVector3d A, B, C;
        double K_i = 2.0 * M_PI, A_i = 0;

        for (int j = 0; j < F1.rows(); j++) {
            Eigen::RowVector3i curr_face = F1.row(j);
            bool found_neighbour = false;
            if (curr_face(0) == i) {
                A = V1.row(curr_face(0));
                B = V1.row(curr_face(1));
                C = V1.row(curr_face(2));
                found_neighbour = true;
            } else if (curr_face(1) == i) {
                A = V1.row(curr_face(1));
                B = V1.row(curr_face(2));
                C = V1.row(curr_face(0));
                found_neighbour = true;
            } else if (curr_face(2) == i) {
                A = V1.row(curr_face(2));
                B = V1.row(curr_face(0));
                C = V1.row(curr_face(1));
                found_neighbour = true;
            }
            if (found_neighbour) {
                double theta = compute_theta(B,A,C);
                double area = 0.5 * (B - A).norm() * (C - A).norm() * sin(theta);
                A_i += area;
                K_i -= theta; // Decrement Angle

            }
        }
        return {A_i, K_i};
    }


} //...ns acq

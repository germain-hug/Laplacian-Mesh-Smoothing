#include "acq/normalEstimation.h"
#include "acq/decoratedCloud.h"
#include "acq/cloudManager.h"
#include "acq/typedefs.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/readOFF.h"
#include "igl/jet.h"
#include "igl/copyleft/cgal/mesh_boolean.h"
#include "igl/copyleft/cgal/intersect_other.h"
#include <math.h>


#include "igl/viewer/Viewer.h"
#include "nanoflann/nanoflann.hpp"
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.hpp"

#include "boost/units/systems/si.hpp"
#include "boost/units/io.hpp"
#include "boost/bind.hpp"

#include <Eigen/Geometry>
#include <Eigen/Sparse>


#include <iostream>

namespace acq {

/** \brief                      Re-estimate normals of cloud \p V fitting planes
 *                              to the \p kNeighbours nearest neighbours of each point.
 * \param[in ] kNeighbours      How many neighbours to use (Typiclaly: 5..15)
 * \param[in ] vertices         Input pointcloud. Nx3, where N is the number of points.
 * \param[in ] maxNeighbourDist Maximum distance between vertex and neighbour.
 * \param[out] viewer           The viewer to show the normals at.
 * \return                      The estimated normals, Nx3.
 */
    NormalsT
    recalcNormals(
            int const kNeighbours,
            CloudT const &vertices,
            float const maxNeighbourDist
    ) {
        NeighboursT const neighbours =
                calculateCloudNeighbours(
                        /* [in]        cloud: */ vertices,
                        /* [in] k-neighbours: */ kNeighbours,
                        /* [in]      maxDist: */ maxNeighbourDist
                );

        // Estimate normals for points in cloud vertices
        NormalsT normals =
                calculateCloudNormals(
                        /* [in]               Cloud: */ vertices,
                        /* [in] Lists of neighbours: */ neighbours
                );

        return normals;
    } //...recalcNormals()

    void setViewerNormals(
            igl::viewer::Viewer &viewer,
            CloudT const &vertices,
            NormalsT const &normals
    ) {
        // [Optional] Set viewer face normals for shading
        //viewer.data.set_normals(normals);

        // Clear visualized lines (see Viewer.clear())
        viewer.data.lines = Eigen::MatrixXd(0, 9);

        // Add normals to viewer
        viewer.data.add_edges(
                /* [in] Edge starting points: */ vertices,
                /* [in]       Edge endpoints: */ vertices + normals * 0.01, // scale normals to 1% length
                /* [in]               Colors: */ Eigen::Vector3d::Zero()
        );
    }

} //...ns acq







void displayMesh(
        igl::viewer::Viewer &viewer,
        acq::CloudManager &cloudManager,
        int const kNeighbours,
        float const maxNeighbourDist,
        Eigen::MatrixXd C
) {

    // ------- First, clear viewer data to avoid conflicts
    viewer.data.clear();

    // ----Retrieve vertices and faces from both meshes
    Eigen::MatrixXd V1 = cloudManager.getCloud(0).getVertices();
    Eigen::MatrixXi F1 = cloudManager.getCloud(0).getFaces();

    // ---- Merge both meshes
    Eigen::MatrixXd V(V1.rows(), V1.cols());
    Eigen::MatrixXi F(F1.rows(), F1.cols());
    cloudManager.setCloud(acq::DecoratedCloud(V1, F1), 0);

    // ------- Show mesh no.1
    viewer.data.set_mesh(V1, F1);
    viewer.data.set_colors(C);

    // Calculate normals on launch
    cloudManager.getCloud(0).setNormals(
            acq::recalcNormals(
                    /* [in]      K-neighbours for FLANN: */ kNeighbours,
                    /* [in]             Vertices matrix: */ cloudManager.getCloud(0).getVertices(),
                    /* [in]      max neighbour distance: */ maxNeighbourDist
            )
    );
    // Update viewer
    acq::setViewerNormals(
            viewer,
            cloudManager.getCloud(0).getVertices(),
            cloudManager.getCloud(0).getNormals()
    );

};


void estimateNormals(
        igl::viewer::Viewer &viewer,
        acq::CloudManager &cloudManager,
        int const kNeighbours,
        float const maxNeighbourDist
) {
    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);
    cloud.setNormals(
            acq::recalcNormals(
                    kNeighbours,
                    cloudManager.getCloud(0).getVertices(),
                    maxNeighbourDist
            )
    );

    cloud.setNormals(acq::recalcNormals(
            kNeighbours,
            cloud.getVertices(),
            maxNeighbourDist
                     )
    );

    acq::NeighboursT const neighbours =
            acq::calculateCloudNeighboursFromFaces(cloud.getFaces());
    cloud.setNormals(
            acq::calculateCloudNormals(cloud.getVertices(), neighbours));

    /// -------- Orient Normals From Faces -------- ///
    int nFlips = acq::orientCloudNormalsFromFaces(cloud.getFaces(), cloud.getNormals());
    acq::setViewerNormals(
            /* [in, out] Viewer to update: */ viewer,
            /* [in]            Pointcloud: */ cloud.getVertices(),
            /* [in] Normals of Pointcloud: */ cloud.getNormals()
    );
}

int main(int argc, char *argv[]) {

    // How many neighbours to use for normal estimation, shown on GUI.
    int kNeighbours = 10;
    // Maximum distance between vertices to be considered neighbours (FLANN mode)
    float maxNeighbourDist = 0.15; //TODO: set to average vertex distance upon read

    // Dummy enum to demo GUI
    enum Orientation {
        Up = 0, Down, Left, Right
    } dir = Up;
    // Dummy variable to demo GUI
    bool boolVariable = true;
    // Dummy variable to demo GUI
    float floatVariable = 0.1f;
    // Noise factor
    double factor = 0.03;

    double lambda = 0.001;

    /// LOAD 3D MODEL
    //std::string meshPath1 = "../../Models/beetle.off";
    std::string meshPath1 = "../../Models/bumpy.off";
    //std::string meshPath1 = "../../Models/cow.off";

    if (argc > 1) {
        meshPath1 = std::string(argv[1]);
        if (meshPath1.find(".off") == std::string::npos) {
            std::cerr << "Only ready for OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh1.off>." << "\n";
    }


    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        viewer.core.show_lines = false;
        viewer.core.show_overlay = true;
    }

    // Store cloud so we can store normals later
    acq::CloudManager cloudManager;
    // Read mesh from meshPath
    {
        Eigen::MatrixXd V1;
        Eigen::MatrixXi F1;

        // Read mesh
        igl::readOFF(meshPath1, V1, F1);

        // Check, if any vertices read
        if (V1.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath1
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        // -------- Store read vertices and faces -------- //
        cloudManager.addCloud(acq::DecoratedCloud(V1, F1));

        Eigen::MatrixXd C(F1.rows(), 3); // Color for display purposes
        C << Eigen::RowVector3d(0.8, 0.6, 0.5).replicate(F1.rows(), 1);
        displayMesh(viewer, cloudManager, kNeighbours, maxNeighbourDist, C);


    } //...read mesh

    // Extend viewer menu using a lambda function
    viewer.callback_init =
            [
                    &cloudManager, &kNeighbours, &maxNeighbourDist,
                    &floatVariable, &boolVariable, &dir, &lambda, &meshPath1, &factor
            ]
                    (igl::viewer::Viewer &viewer) {

                // Add an additional menu window
                viewer.ngui->addGroup("Mesh");
                viewer.ngui->addWindow(Eigen::Vector2i(900, 10), "Coursework 2");
                viewer.ngui->addButton("Reset Mesh", [&]() {
                    Eigen::MatrixXd V1;
                    Eigen::MatrixXi F1;
                    if(dir==0) {
                        meshPath1 = "../../Models/bumpy.off";
                    } else if(dir == 1){
                        meshPath1 = "../../Models/cow.off";
                    }

                    igl::readOFF(meshPath1, V1, F1);
                    cloudManager.getCloud(0).setVertices(V1);
                    cloudManager.getCloud(0).setFaces(F1);
                    Eigen::MatrixXd C(F1.rows(), 3); // Color for display purposes
                    C << Eigen::RowVector3d(0.8, 0.6, 0.5).replicate(F1.rows(), 1);
                    displayMesh(viewer, cloudManager, kNeighbours, maxNeighbourDist, C);
                });

                viewer.ngui->addVariable<Orientation>("Mesh", dir)->setItems(
                        {"Bumpy","Cow"}
                );

                viewer.ngui->addGroup("Discrete Curvature");

                /**  ======================================== **/
                /**  ========== I) a) MEAN CURVATURE ======== */
                /**  ======================================== **/

                viewer.ngui->addButton("Mean Curvature H", [&]() {
                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                    /// -------- Estimate Normals From Faces -------- ///
                    estimateNormals(viewer, cloudManager, kNeighbours, maxNeighbourDist);

                    /// -------- Call Mean Curvature -------- ///
                    Eigen::VectorXd H = cloud.meanCurvature();

                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    igl::jet(H, true, C);
                    viewer.data.set_colors(C);
                });


                /**  =========================================== **/
                /**  ========= I) b) GAUSSIAN CURVATURE ======== **/
                /**  =========================================== **/


                viewer.ngui->addButton("Gaussian Curvature K", [&]() {

                    /// -------- Call Mean Curvature -------- ///
                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);
                    Eigen::VectorXd K = cloud.gaussianCurvature();

                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    igl::jet(K, true, C);
                    viewer.data.set_colors(C);
                });


                /**  ========================================= **/
                /**  ==== II) DISCRETE LAPLACE BELTRAMI ====== **/
                /**  ========================================= **/

                viewer.ngui->addButton("Discrete Mean Curvature", [&]() {
                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                    /// -------- Estimate Normals From Faces -------- ///
                    estimateNormals(viewer, cloudManager, kNeighbours, maxNeighbourDist);

                    /// ------- Retrieve Adjacency Matrices
                    Eigen::MatrixXi F = cloud.getFaces();
                    Eigen::MatrixXi TT, TTi;
                    igl::triangle_triangle_adjacency(F, TT, TTi);

                    /// -------- Call Mean Curvature -------- ///
                    Eigen::VectorXd H = cloud.discreteMeanCurvature(TT, TTi,lambda, false, false);


                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    igl::jet(H, true, C);
                    viewer.data.set_colors(C);
                });





                /**  ============================================= **/
                /**  ==== III) EXPLICIT LAPLACIAN SMOOTHING ====== **/
                /**  ============================================= **/

                viewer.ngui->addGroup("Laplacian Mesh Smoothing");
                viewer.ngui->addButton("Explicit Smoothing", [&]() {

                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                    /// -------- Estimate Normals From Faces -------- ///
                    estimateNormals(viewer, cloudManager, kNeighbours, maxNeighbourDist);

                    /// ------- Retrieve Adjacency Matrices
                    Eigen::MatrixXi F = cloud.getFaces();
                    Eigen::MatrixXi TT, TTi;
                    igl::triangle_triangle_adjacency(F, TT, TTi);

                    /// -------- Compute Mean Curvature -------- ///
                    Eigen::VectorXd H = cloud.discreteMeanCurvature(TT, TTi,lambda, true, false);


                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    igl::jet(H, true, C);
                    viewer.data.set_colors(C);
                    displayMesh(viewer,cloudManager,kNeighbours,maxNeighbourDist,C);

                });


                /**  ============================================= **/
                /**  ===== IV) IMPLICIT LAPLACIAN SMOOTHING ====== **/
                /**  ============================================= **/

                viewer.ngui->addGroup("Laplacian Mesh Smoothing");
                viewer.ngui->addButton("Implicit Smoothing", [&]() {

                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                    /// -------- Estimate Normals From Faces -------- ///
                    estimateNormals(viewer, cloudManager, kNeighbours, maxNeighbourDist);

                    /// ------- Retrieve Adjacency Matrices
                    Eigen::MatrixXi F = cloud.getFaces();
                    Eigen::MatrixXi TT, TTi;
                    igl::triangle_triangle_adjacency(F, TT, TTi);

                    /// -------- Compute Mean Curvature -------- ///
                    Eigen::VectorXd H = cloud.discreteMeanCurvature(TT, TTi,lambda, false, true);


                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    igl::jet(H, true, C);
                    viewer.data.set_colors(C);
                    displayMesh(viewer,cloudManager,kNeighbours,maxNeighbourDist,C);

                });

                viewer.ngui->addVariable("Lambda", lambda);


                /**  ========================= **/
                /**  ===== V) DENOISING ====== **/
                /**  ========================= **/

                viewer.ngui->addGroup("Perturb Mesh");
                viewer.ngui->addVariable("Noise Factor", factor);
                viewer.ngui->addButton("Add Noise", [&]() {

                    acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                    /// ----- Add Noise To Mesh ---- ///
                    cloud.addNoise(factor);

                    /// -------- Display Mesh -------- ///
                    Eigen::MatrixXd C(cloud.getFaces().rows(), 3);
                    C << Eigen::RowVector3d(0.8, 0.6, 0.5).replicate(
                            cloud.getFaces().rows(), 1);
                    displayMesh(viewer, cloudManager, kNeighbours, maxNeighbourDist, C);

                });



                // Generate menu
                viewer.screen->performLayout();

                return false;
            }; //...viewer menu



    // Start viewer
    viewer.launch();

    return 0;
} //...main()

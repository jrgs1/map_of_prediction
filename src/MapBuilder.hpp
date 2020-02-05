#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <ros/console.h>
#include <fstream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

ofstream file1("plik.csv");
ofstream file2("predykcja_w_przod.csv");

class MapBuilder
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    const int kinect_cols = 640;        //liczba pikseli kinecta wzdłuż osi X
    const float min_range = 0.5;        //początkowy zakres pomiaru kinecta: 0,5 m

    bool show_windows;
    bool write_to_files;

    int map_y_size;                     //liczba komórek na mapie w osi Y (wierszy), jest obliczana z zakresu i wielkości komórki
    int map_x_size;                     //liczba komórek na mapie w osi X (kolumn). Podawana przez użytkownika przy deklaracji obiektu
    int prediction_map_x_size;          //rozmiar mapy predykcji
    int prediction_map_y_size;
    long long int iteration_counter;    //licznik wykonań funkcji callback subskrybenta. Służy do określania ilości zrzutu danych do pliku csv
    long long int iteration_counter_2;    //licznik wykonań funkcji callback subskrybenta. Służy do określania ilości zrzutu danych do pliku csv
	int velocity_vector_length;         //wielkość wektora zdykretyzowanych prędkości
    float occupancy_initial_value;           //wartość komórki mapy pomiarów, gdy wykryto przeszkodę
    float max_range;                    //maksymalny zasięg pomiarów w metrach. Podawany przez użytkownika
    float bayes_normalizer_estimation;  //normalizator estymacji
    float bayes_normalizer_prediction;  //normalizator predykcji bayesowskiej
    float epsilon;                      //wartość bliska zeru, wypełnia puste komórki
    float cell_y_size;                  //rozmiar komórki w osi Y. Podawany przez użytkownika
	float robot_rotation_angle;         //wartości dodatnie to obrót w lewo, ujemne - w prawo
	float blur_coeff;                   //współczynnik rozmycia predykcji
	Mat current_map, past_map;          //bieżąca i poprzednia mapa środowiska
    Mat velocities_x, velocities_y, velocities_xy, velocities_xy_dir_flag;    //mapa prędkości w osiach X i Y, X+Y, mapa kierunków prędkości dla X=Y

    vector<Mat> prediction_of_future_positions;    //mapa przyszłych położeń przeszkód w kolejnych chwilach
	vector<float> V;                    //wektor zdyskretyzowanych prędkości (jednostka: komórki na sekundę)


public:
    MapBuilder(float cell_size_in_y_in_meters, float kinect_max_range_in_y, int seconds_of_prediction, float blur_kernel_coeff, bool show_map_on_screen, bool write_maps_to_file);

    ~MapBuilder();

private:
    void write_matrix_to_file(Mat matrix, ofstream & file);//zapisuje mapy do plików csv

    Mat rotate_matrix(Mat _input_matrix, float angle);   //obraca macierz o zadany kąt

    static void show_image(const Mat& _image, const string& _winname);  //normalizacja obrazu przed wyświetleniem

    Mat project_depth_image_to_plane(Mat _input_image);  //rzut mapy glębii na płaszczyznę

    void bayesian_prediction();

    void calculate_velocities();    //wykrywanie ruchu i szacowanie prędkości

    void bayesian_estimation(const Mat& _measurement);

    void predict_future_positions();    //przewidywanie przyszłych położeń komórek

    void place_static_obstacles(Mat _vector);    //umieszczenie nieruchomych obiektów na mapie

    Mat make_test_measurements();   //generowanie testowych pomiarów

    void earse_cells_behind_obstacle(Mat _measurement);  //czyszczenie komórek zasłoniętych przez przeszkody

    void imageCb(const sensor_msgs::ImageConstPtr& msg);

};




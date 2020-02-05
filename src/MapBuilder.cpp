#include "MapBuilder.hpp"

using namespace cv;
using namespace std;


	MapBuilder::MapBuilder(float cell_size_in_y_in_meters, float kinect_max_range_in_y, int seconds_of_prediction, float blur_kernel_coeff, bool show_map_on_screen, bool write_maps_to_file)
    : it_(nh_)
    {
		image_sub_ = it_.subscribe("/camera/depth/image", 1, &MapBuilder::imageCb, this);
		image_pub_ = it_.advertise("/map/prediction", 1);
        iteration_counter = 0;
        iteration_counter_2 = 0;
        epsilon = 0.01;
        occupancy_initial_value = 1;
        bayes_normalizer_estimation = 0;
        bayes_normalizer_prediction = 0;
        cell_y_size = cell_size_in_y_in_meters;
        max_range = kinect_max_range_in_y;
		map_y_size =  int((max_range - min_range)/cell_y_size);
	    map_x_size = map_y_size;
	    prediction_map_x_size = map_x_size*3;
	    prediction_map_y_size = map_y_size*3;
		robot_rotation_angle = 20;
		velocity_vector_length = 10;
		blur_coeff = blur_kernel_coeff;
		show_windows = show_map_on_screen;
		write_to_files = write_maps_to_file;

	    current_map = Mat::ones(map_y_size, map_x_size, CV_32F);
		current_map *= 0.5;
		past_map = Mat::ones(map_y_size, map_x_size, CV_32F);
		past_map *= 0.5;

		//utworzenie wektora map przyszłych położeń
        for(int i=0; i<seconds_of_prediction+1; i++)
        {
            prediction_of_future_positions.push_back(Mat::zeros(prediction_map_y_size, prediction_map_x_size, CV_32F));
        }

        //utowrzenie wektora z wartościami prędkości
        for(int i = 0; i < velocity_vector_length; i++)
        {
	        V.push_back(0);
        }

	    float vel_temp = -floor(float(velocity_vector_length)/2);
	    int j=0;
        for(int i = 0; i < V.size(); i++)
        {
        	if(i < V.size()/2)
	            V[i] = -(i+1);
	        else
	        {
		        V[i] = i-j;
		        j+=2;
	        }
        }

        velocities_y = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_x = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_xy = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_xy_dir_flag = Mat::zeros(map_y_size, map_x_size, CV_32F);

		namedWindow("Mapa", WINDOW_FREERATIO);
		namedWindow("Prediction in 1 sec", WINDOW_FREERATIO);
		namedWindow("Pomiar", WINDOW_FREERATIO);

    }

	MapBuilder::~MapBuilder()
    {
		destroyAllWindows();
		file1.close();
		file2.close();
    }


    void MapBuilder::write_matrix_to_file(Mat matrix, ofstream & file) //zapisuje mapy do plików csv
    {

        file << endl << endl;
        file << "nr: \t" << iteration_counter << endl;
        for(int row_count = 0; row_count < matrix.rows; row_count++)
        {
            file << row_count << ": ";
            for(int col_count = 0; col_count < matrix.cols; col_count++)
            {
                file << matrix.at<float>(row_count, col_count)<< "\t";
            }
            file << endl;
        }
    }

    Mat MapBuilder::rotate_matrix(Mat _input_matrix, float angle)   //obraca macierz o zadany kąt
    {
	    Mat rotation_matrix = getRotationMatrix2D(Point2f(_input_matrix.rows-1, int(_input_matrix.cols/2)), angle, 1);
	    Mat _output_matrix;
	    warpAffine(_input_matrix, _output_matrix, rotation_matrix, _input_matrix.size(), INTER_NEAREST);
	    return _output_matrix;

    }

    void MapBuilder::show_image(const Mat& _image, const string& _winname)  //normalizacja obrazu przed wyświetleniem
    {
        Mat output_image;
        normalize(_image, output_image, 0, 65535, NORM_MINMAX, CV_16UC1);
        imshow(_winname, output_image);
    }

    Mat MapBuilder::project_depth_image_to_plane(Mat _input_image)  //rzut mapy glębii na płaszczyznę
    {
        //inicjalizacja wstępnej mapy o rozmiarze (ilość komórek mapy * rozdzielczość kinecta)
        Mat img_map = Mat::zeros(map_y_size,kinect_cols, CV_32F);
        Mat _output_image;

        //zerowanie pikseli od 1/3 wysokości (żeby nie łapać lampy i sufitu)
        for(int row_count=0; row_count < _input_image.rows/3; row_count++)
            for(int col_count=0; col_count < _input_image.cols; col_count++)
                _input_image.at<float>(row_count, col_count) = 0;

        Mat img_bin = _input_image.clone();
        img_bin = Mat::zeros(_input_image.rows, _input_image.cols, CV_8U);

        float k = min_range;
        for(int i=0; i<map_y_size; i++)
        {
            k += cell_y_size;
            inRange(_input_image, k, k+cell_y_size, img_bin);
            Mat el;
            el = Mat::ones(9,9,CV_8S);
            morphologyEx(img_bin, img_bin, MORPH_CLOSE, el, Point(-1, -1), 4);
            vector<vector<Point> > contours;
            vector<Point> contours_poly;
            Rect boundRect;
            findContours( img_bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            for(const auto & contour : contours)
            {
                approxPolyDP( Mat(contour), contours_poly, 5, true );
                boundRect = boundingRect( Mat(contours_poly) );
                rectangle( img_bin, boundRect.tl(), boundRect.br(), 255, 2, 8, 0 );
                for(int j=0; j<img_bin.cols; j++)
                {
                    if(j >= boundRect.x && j <= (boundRect.x+boundRect.width))
                        img_map.at<float>(i,j)=occupancy_initial_value;
                }
            }

        }

        //przeskalowanie mapy w osi x, żeby komórka miała większy rozmiar i by było ich mniej na mapie
        resize(img_map, _output_image, Size(map_x_size, map_y_size), 0, 0, INTER_NEAREST);

        //odbicie macierzy wzlgędem osi x
        flip(_output_image, _output_image, 0);

        //zerowanie pierwszego wiersza, by skrocic szukanie elementu minimalnego dla kazdej kolumny
        for(int col_count = 0; col_count < _output_image.cols; col_count++)
            _output_image.at<float>(0, col_count) = 0;

        //zerowanie komórek za przeszkodą
        earse_cells_behind_obstacle(_output_image);

        return _output_image;
    }

    void MapBuilder::bayesian_prediction()
    {
        Mat _velocities_temp, _velocities_aggregated;
        add(velocities_y, velocities_x, _velocities_temp);
        add(_velocities_temp, velocities_xy, _velocities_aggregated);
        for(int col_count = 0; col_count < past_map.cols; col_count++)
        {
            for(int row_count = 0; row_count < past_map.rows; row_count++)
            {
                past_map.at<float>(row_count, col_count) = (_velocities_aggregated.at<float>(row_count, col_count)
                                                              + past_map.at<float>(row_count, col_count));     //licznik estymacji - pomiar razy wartośc z poprzedniej chwili
                if(past_map.at<float>(row_count, col_count) == 0)
                    past_map.at<float>(row_count, col_count) = epsilon;     //przypisanie zerowym komórkom małej wartości, by przy pojawieniu się przeszkody nie mnożyć przez zero. Wartosc większa niż 0.01 destablizuje obliczenia
                bayes_normalizer_prediction += past_map.at<float>(row_count, col_count); //obliczanie normalizatora
            }


            for(int row_count = 0; row_count < past_map.rows; row_count++)
            {
                past_map.at<float>(row_count, col_count) /= bayes_normalizer_prediction;  //mianownik estymacji - dzielenie dzielenie przez normalizator
            }
            bayes_normalizer_prediction = 0;
        }
    }

    void MapBuilder::calculate_velocities()     //wykrywanie ruchu i szacowanie prędkości
    {

        for(int col_count = 1; col_count < current_map.cols-1; col_count++)
        {
            for (int row_count = 1; row_count < current_map.rows-1; row_count++)
            {
	            //prędkości w kierunku Y
                if(current_map.at<float>(row_count, col_count) != current_map.at<float>(0, col_count) && past_map.at<float>(row_count, col_count) == past_map.at<float>(0, col_count))
                {
                    float velocity_temp = 0;
                    if(past_map.at<float>(row_count - 1, col_count) != past_map.at<float>(0, col_count))
                        velocity_temp = -fabs(past_map.at<float>(row_count - 1, col_count) - current_map.at<float>(row_count, col_count));
                    if(past_map.at<float>(row_count + 1, col_count) != past_map.at<float>(0, col_count))
                        velocity_temp = fabs(past_map.at<float>(row_count +1, col_count) - current_map.at<float>(row_count, col_count));

                    int cntr = 0;
                    for(int i = -ceil((V.size()+1)/2); i < ceil((V.size()+1)/2); i++)
                    {
                    	float p_low = 2*float(i)/V.size();
                    	float p_high = p_low+2*float(1)/V.size();

                        if(p_high == 0)
                        	p_high = -epsilon;
                        if(velocity_temp > p_low && velocity_temp <= p_high)
	                        velocities_y.at<float>(row_count, col_count) = V[cntr];
                        cntr++;

                    }
                }
                else
                    velocities_y.at<float>(row_count, col_count) = 0;

                //prędkości w kierunku X
	            if(current_map.at<float>(row_count, col_count) != current_map.at<float>(0, col_count) && past_map.at<float>(row_count, col_count) == past_map.at<float>(0, col_count))
	            {
		            float velocity_temp = 0;
		            if(past_map.at<float>(row_count, col_count - 1) != past_map.at<float>(0, col_count-1))
			            velocity_temp = fabs(past_map.at<float>(row_count, col_count - 1) - current_map.at<float>(row_count, col_count));
		            if(past_map.at<float>(row_count, col_count + 1) != past_map.at<float>(0, col_count+1))
			            velocity_temp = -fabs(past_map.at<float>(row_count, col_count +1) - current_map.at<float>(row_count, col_count));

		            int cntr = 0;
		            for(int i = -ceil((V.size()+1)/2); i < ceil((V.size()+1)/2); i++)
		            {
			            float p_low = 2*float(i)/V.size();
			            float p_high = p_low+2*float(1)/V.size();

			            if(p_high == 0)
				            p_high = -epsilon;
			            if(velocity_temp > p_low && velocity_temp <= p_high)
				            velocities_x.at<float>(row_count, col_count) = V[cntr];
			            cntr++;

		            }
	            }
	            else
		            velocities_x.at<float>(row_count, col_count) = 0;// temp_velocity_aggregated_x;//*velocities_x[1].at<float>(row_count, col_count);

	            //prędkości w kierunku XY
		        if(current_map.at<float>(row_count, col_count) != current_map.at<float>(0, col_count) && past_map.at<float>(row_count, col_count) == past_map.at<float>(0, col_count) /*&& velocities_y.at<float>(row_count, col_count) == 0*/)
	            {
		            float velocity_temp = 0;
		            if(past_map.at<float>(row_count-1, col_count - 1) != past_map.at<float>(0, col_count-1))
		            {
			            velocity_temp = fabs(past_map.at<float>(row_count-1, col_count - 1) - current_map.at<float>(row_count, col_count));
			            velocities_xy_dir_flag.at<float>(row_count, col_count) = 1;
		            }
		            if(past_map.at<float>(row_count-1, col_count + 1) != past_map.at<float>(0, col_count+1))
		            {
			            velocity_temp = fabs(past_map.at<float>(row_count-1, col_count +1) - current_map.at<float>(row_count, col_count));
			            velocities_xy_dir_flag.at<float>(row_count, col_count) = 2;
		            }
		            if(past_map.at<float>(row_count+1, col_count - 1) != past_map.at<float>(0, col_count-1))
		            {
			            velocity_temp = fabs(past_map.at<float>(row_count+1, col_count - 1) - current_map.at<float>(row_count, col_count));
			            velocities_xy_dir_flag.at<float>(row_count, col_count) = 3;
		            }
		            if(past_map.at<float>(row_count+1, col_count + 1) != past_map.at<float>(0, col_count+1))
		            {
			            velocity_temp = fabs(past_map.at<float>(row_count+1, col_count +1) - current_map.at<float>(row_count, col_count));
			            velocities_xy_dir_flag.at<float>(row_count, col_count) = 4;
		            }
		            int cntr = 0;
		            for(int i = -ceil((V.size()+1)/2); i < ceil((V.size()+1)/2); i++)
		            {
			            float p_low = 2*float(i)/V.size();
			            float p_high = p_low+2*float(1)/V.size();

			            if(p_high == 0)
				            p_high = -epsilon;
			            if(velocity_temp > p_low && velocity_temp <= p_high)
				            velocities_xy.at<float>(row_count, col_count) = fabs(V[cntr]);
			            cntr++;

		            }
	            }else
		            velocities_xy.at<float>(row_count, col_count) = 0;// temp_velocity_aggregated_x;//*velocities_x[1].at<float>(row_count, col_count);
            }
        }
    }

    void MapBuilder::bayesian_estimation(const Mat& _measurement)
    {
        for(int col_count = 0; col_count < current_map.cols; col_count++)
        {
            for(int row_count = 0; row_count < current_map.rows; row_count++)
            {
                current_map.at<float>(row_count, col_count) = (_measurement.at<float>(row_count, col_count)
                                                              * past_map.at<float>(row_count, col_count));     //licznik estymacji - pomiar razy wartośc z poprzedniej chwili
                if(current_map.at<float>(row_count, col_count) == 0)
                    current_map.at<float>(row_count, col_count) = epsilon;     //przypisanie zerowym komórkom małej wartości, by przy pojawieniu się przeszkody nie mnożyć przez zero. Wartosc większa niż 0.01 destablizuje obliczenia
                bayes_normalizer_estimation += current_map.at<float>(row_count, col_count); //obliczanie normalizatora
            }


            for(int row_count = 0; row_count < current_map.rows; row_count++)
            {
                current_map.at<float>(row_count, col_count) /= bayes_normalizer_estimation;  //mianownik estymacji - dzielenie dzielenie przez normalizator
            }
            bayes_normalizer_estimation = 0;
        }
    }

    void MapBuilder::predict_future_positions()     //przewidywanie przyszłych położeń komórek
    {
    	int shift_x = (prediction_of_future_positions[0].cols - current_map.rows)/2;
    	int shift_y = prediction_of_future_positions[0].rows/2 - current_map.rows;
        for(int col_count = 0; col_count < velocities_y.cols; col_count++)
        {
            for (int row_count = 0; row_count < velocities_y.rows; row_count++)
            {
                //rozkład położeń wprzód - X
                if(velocities_y.at<float>(row_count, col_count) == 0 && velocities_x.at<float>(row_count, col_count) != 0 )
                {
                    if(velocities_x.at<float>(row_count, col_count) > 0)
                    {
                        for(int time_step = 1; time_step < prediction_of_future_positions.size(); time_step++)
                        {
                            int s = floor(fabs(velocities_x.at<float>(row_count, col_count))*float(time_step));
                            if((col_count+shift_x+s) <= prediction_of_future_positions[time_step].cols)
                                prediction_of_future_positions[time_step].at<float>(row_count+shift_y, col_count+shift_x+s) = 1;
                        }
                    }

                    if(velocities_x.at<float>(row_count, col_count) < 0)
                    {
                        for(int time_step = 1; time_step < prediction_of_future_positions.size(); time_step++)
                        {
                            int s = floor(fabs(velocities_x.at<float>(row_count, col_count))*float(time_step));
                            if((col_count+shift_x-s) >= 0)
                                prediction_of_future_positions[time_step].at<float>(row_count+shift_y, col_count+shift_x-s) = 1;
                        }
                    }
                }

                //rozkład położeń wprzód - Y
                if(velocities_x.at<float>(row_count, col_count) == 0 && velocities_y.at<float>(row_count, col_count) != 0)
                {
                    if(velocities_y.at<float>(row_count, col_count) < 0)
                    {
                        for(int time_step = 1; time_step < prediction_of_future_positions.size(); time_step++)
                        {
                            int s = floor(fabs(velocities_y.at<float>(row_count, col_count))*float(time_step));
                            if((row_count+shift_y+s) <= prediction_of_future_positions[time_step].rows)
                                prediction_of_future_positions[time_step].at<float>(row_count+shift_y+s, col_count+shift_x) = 1;
                        }
                    }

                    if(velocities_y.at<float>(row_count, col_count) > 0)
                    {
                        for(int time_step = 1; time_step < prediction_of_future_positions.size(); time_step++)
                        {
                            int s = floor(fabs(velocities_y.at<float>(row_count, col_count))*float(time_step));
                            if((row_count+shift_y-s) >= 0)
                                prediction_of_future_positions[time_step].at<float>(row_count+shift_y-s, col_count+shift_x) = 1;
                        }
                    }
                }

                //rozkład położeń wprzód - XY (kątowo)
                if(velocities_xy.at<float>(row_count, col_count) != 0)
                {
	                for(int time_step = 1; time_step < prediction_of_future_positions.size(); time_step++)
	                {
		                int s = floor(fabs(velocities_xy.at<float>(row_count, col_count))*float(time_step));
		                if(velocities_xy_dir_flag.at<float>(row_count, col_count) == 1)
		                {
		                	if((row_count+shift_y+s) < prediction_of_future_positions[time_step].rows && col_count+shift_x+s < prediction_of_future_positions[time_step].cols)
		                	    prediction_of_future_positions[time_step].at<float>(row_count+shift_y+s, col_count+shift_x+s) = 1;
		                }
		                if(velocities_xy_dir_flag.at<float>(row_count, col_count) == 2)
		                {
			                if((row_count+shift_y+s) < prediction_of_future_positions[time_step].rows && col_count+shift_x-s >= 0)
				                prediction_of_future_positions[time_step].at<float>(row_count+shift_y+s, col_count+shift_x-s) = 1;
		                }
		                if(velocities_xy_dir_flag.at<float>(row_count, col_count) == 3)
		                {
			                if((row_count+shift_y-s) >= 0 && col_count+shift_x+s < prediction_of_future_positions[time_step].cols)
				                prediction_of_future_positions[time_step].at<float>(row_count+shift_y-s, col_count+shift_x+s) = 1;
		                }
		                if(velocities_xy_dir_flag.at<float>(row_count, col_count) == 4)
		                {
			                if((row_count+shift_y-s) >= 0 && col_count+shift_x-s >= 0)
				                prediction_of_future_positions[time_step].at<float>(row_count+shift_y-s, col_count+shift_x-s) = 1;
		                }
	                }
                }
            }
        }

        for(int i = 1; i < prediction_of_future_positions.size(); i++)
        {
            int krnl = floor(i*2+1);
            GaussianBlur(prediction_of_future_positions[i], prediction_of_future_positions[i], Size(krnl, krnl), 0);
        }
    }

    void MapBuilder::place_static_obstacles(Mat _vector)    //umieszczenie nieruchomych obiektów na mapie
    {
        Rect rect((_vector.cols-current_map.cols)/2, _vector.rows/2-current_map.rows, current_map.cols, current_map.rows);
        //current_map.copyTo(_vector(rect));
        add(current_map, _vector(rect), _vector(rect));
           // add(current_map, _vector[i], _vector[i]);
            //zerowanie komórek, które już się przesunęły
            for(int col_count = 0; col_count < velocities_y.cols; col_count++)
            {
                for (int row_count = 0; row_count < velocities_y.rows; row_count++)
                {
                    if(velocities_y.at<float>(row_count, col_count) != 0 || velocities_x.at<float>(row_count, col_count) != 0 || velocities_xy.at<float>(row_count, col_count) != 0)
	                    _vector.at<float>(row_count, col_count) = 0;
                }
            }

    }

    Mat MapBuilder::make_test_measurements()    //generowanie testowych pomiarów
    {
	    Mat test_measurements = Mat::zeros(map_x_size, map_y_size, CV_32F);
	    float circle_movement = 0.8*iteration_counter_2;
	    float circle_movement2 = 0.7*iteration_counter_2;
	    //circle(test_measurements, Point(test_measurements.rows - 3 - circle_movement, test_measurements.cols/2), 1, 255);
	    //circle(test_measurements, Point(test_measurements.rows/2, test_measurements.cols - 3 - circle_movement), 1, 255);
	    //circle(test_measurements, Point(3 + circle_movement, test_measurements.rows - 3 - circle_movement2), 3, 255, 2);
	    //circle(test_measurements, Point(test_measurements.rows - 2, test_measurements.cols -2), 1, 255);

	    //rectangle(test_measurements, Point(circle_movement, test_measurements.rows/2), Point(circle_movement+3,test_measurements.rows/2), 255);
	    //rectangle(test_measurements, Point(test_measurements.cols/2, circle_movement), Point(test_measurements.cols/2+3,circle_movement), 255);
	    rectangle(test_measurements, Point(test_measurements.cols - 5 - circle_movement, circle_movement-1), Point(test_measurements.cols - circle_movement, circle_movement+1), 255);
	    //rectangle(test_measurements, Point(circle_movement, test_measurements.rows - circle_movement), Point(circle_movement+5, test_measurements.rows - circle_movement), 255);

	    if(int(test_measurements.rows-5-circle_movement) == 0 || int(test_measurements.cols-5-circle_movement) == 0)
		    iteration_counter_2 = 0;
	    earse_cells_behind_obstacle(test_measurements);
	    return test_measurements;
    }

    void MapBuilder::earse_cells_behind_obstacle(Mat _measurement)  //czyszczenie komórek zasłoniętych przez przeszkody
    {
        bool has_obstacle = false;
        for(int col_count=_measurement.cols; col_count >= 0; col_count--)
        {
            for(int row_count=_measurement.rows-1; row_count >= 0 ; row_count--)
            {

                if(_measurement.at<float>(row_count+1, col_count) >= occupancy_initial_value)
                    has_obstacle = true;
                if(has_obstacle)
                    _measurement.at<float>(row_count, col_count) = 0;
            }
            has_obstacle = false;
        }
    }

    void MapBuilder::imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
		try
        {
			cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
		catch (cv_bridge::Exception& e)
        {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
        }

		//przepisanie przechywconej klatki do obiektu Mat
		Mat raw_point_cloud = cv_ptr->image;

		//rzut punktów na płaszczyznę
        Mat measurement = project_depth_image_to_plane(raw_point_cloud);

        //obrót pomiarów
        //measurement = rotate_matrix(measurement, robot_rotation_angle);

        //pomiary testowe
        //make_test_measurements().copyTo(measurement);

		//predykcja1
	    bayesian_prediction();

        //estymacja
        bayesian_estimation(measurement);

        //obliczenie prędkości
        calculate_velocities();

        //predykcja2
        bayesian_prediction();

        //przyszłe położenia
        predict_future_positions();

        //umieszczenie statycznych przeszkód na mapie przewidywań położeń
        for(int i = 1; i < prediction_of_future_positions.size(); i++)
            place_static_obstacles(prediction_of_future_positions[i]);

        //zapamiętanie poprzednich map
        current_map.copyTo(past_map);

        if(show_windows)
        {
	        show_image(current_map, "Mapa");
	        show_image(measurement, "Pomiar");
	        show_image(prediction_of_future_positions[1], "Prediction in 1 sec");
	        waitKey(5); //opóżnienie potrzebne do wyświetlania map
        }


        //publikowanie mapy z przewidywaniami
        sensor_msgs::ImagePtr out_image = cv_bridge::CvImage(std_msgs::Header(), "mono16", prediction_of_future_positions[1]).toImageMsg();
        image_pub_.publish(out_image);

        //licznik iteracji
        iteration_counter++;
        iteration_counter_2++;

        //zapisywanie pierwszych map do plików csv
        if(write_to_files && iteration_counter >= 0 && iteration_counter < 80)
        {
            write_matrix_to_file(velocities_y, file1);
            write_matrix_to_file(current_map, file2);
        }

        //kasowanie zmiennych
        velocities_y = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_x = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_xy = Mat::zeros(map_y_size, map_x_size, CV_32F);
        velocities_xy_dir_flag = Mat::zeros(map_y_size, map_x_size, CV_32F);
        for(int i = 0; i < prediction_of_future_positions.size(); i++)
        {
            prediction_of_future_positions[i] = Mat::zeros(prediction_map_y_size, prediction_map_x_size, CV_32F);
        }

    }



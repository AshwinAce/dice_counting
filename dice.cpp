#include<iostream>
#include<string>
#include<sstream>
#include<sys/stat.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

void display_details(int number_of_labels, cv::Mat centroids, cv::Mat stats);
void get_number_of_dice(int number_of_labels,int inv_number_of_labels, int counts_of_dice[], cv::Mat stats, cv::Mat inv_stats, cv::Mat centroids, int min_pixels, int max_pixels);
void write_counts(int inv_number_of_labels, int counts_of_dice[], cv::Mat img, cv::Mat inv_stats);
void create_borders(cv::Mat img, cv::Mat border);

int main(int argc,char **argv){
	struct stat s;
	if(stat(argv[1],&s)==0){
		if(!(s.st_mode & S_IFDIR)){
			std::cout<<"First parameter needs to be output folder to save results\n";
			return EXIT_FAILURE;
		}
	}
	for(int i=2;i<argc;i++){
		try{
			cv::Mat img=cv::imread(argv[i],1);
			cv::Mat img_gray,mask,inv_mask,border;
			cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
			/*For connected components on thresholded image and its binary inverse. Gets regions of dices as well as individual dots between these two images.*/ 
			cv::Mat labels,stats,centroids,inv_labels,inv_stats,inv_centroids;
			cv::threshold(img_gray,mask,180,255,cv::THRESH_BINARY_INV);
			/*Dilation to remove noise*/
			cv::morphologyEx(mask,mask,cv::MORPH_DILATE,cv::Mat(),cv::Point(-1,-1),1,1,1);
			cv::threshold(mask,inv_mask,180,255,cv::THRESH_BINARY_INV);
			
			/*Gradient is used to draw green borders around dice*/
			cv::morphologyEx(mask,border,cv::MORPH_GRADIENT,cv::Mat(),cv::Point(-1,-1),1,1,1);
			cv::threshold(border,border,180,255,cv::THRESH_BINARY_INV);
		
			int not_required,number_of_labels= cv::connectedComponentsWithStats(mask,labels,stats,centroids);
			int not_required1,inv_number_of_labels= cv::connectedComponentsWithStats(inv_mask,inv_labels,inv_stats,inv_centroids);
			/*Gets number of dots and dices*/
			int counts_of_dice[inv_number_of_labels]={0};
			int min_pixels=350,max_pixels=1000;
			get_number_of_dice(number_of_labels,inv_number_of_labels, counts_of_dice, stats, inv_stats, centroids, min_pixels, max_pixels);

			/*Writing the counts on the image*/
			write_counts(inv_number_of_labels, counts_of_dice, img, inv_stats);
			/*Adding green border around the dice and the dots*/
			create_borders(img, border);

			cv::imshow("Img",img);
			cv::waitKey(1000);
			cv::destroyAllWindows();
			/*Adding output to name in case source and destination folders of images are the same*/
			std::string filename=std::string(argv[1])+"/output_"+std::string(argv[i]);
			cv::imwrite(filename,img);
		}
		catch(cv::Exception& e){
			std::cout<<"Exception"<<e.what()<<std::endl;
		}
	}
	return EXIT_SUCCESS;
}

/* A function for displaying the region boundaries as well the total pixels in it. Not necessary ,but can be useful for debugging*/
void display_details(int number_of_labels, cv::Mat centroids, cv::Mat stats){
	for (int i=0;i<number_of_labels;i++){
		std::cout<<centroids.at<double>(i,0)<<centroids.at<double>(i,1)<<std::endl;
		std::cout<<"Region from "<<stats.at<int>(i,cv::CC_STAT_LEFT)<<" to "<<stats.at<int>(i,cv::CC_STAT_LEFT)+stats.at<int>(i,cv::CC_STAT_WIDTH)<<std::endl;
		std::cout<<"Region from "<<stats.at<int>(i,cv::CC_STAT_TOP)<<" to "<<stats.at<int>(i,cv::CC_STAT_TOP)+stats.at<int>(i,cv::CC_STAT_HEIGHT)<<std::endl;
		std::cout<<"Area is "<<stats.at<int>(i,cv::CC_STAT_AREA)<<" pixels."<<std::endl;
	}
	std::cout<<"Number of labels is "<<number_of_labels<<std::endl;
}
/* A function that returns the number of dots in each dice*/ 
void get_number_of_dice(int number_of_labels,int inv_number_of_labels, int counts_of_dice[], cv::Mat stats, cv::Mat inv_stats, cv::Mat centroids, int min_pixels, int max_pixels){
	for (int i=1;i<number_of_labels;i++){
		for (int j=1;j<inv_number_of_labels;j++){
			int x=centroids.at<double>(i,0);
			int y=centroids.at<double>(i,1);
			bool x_inside, y_inside;
			/*checking if centroids of the dots are inside the dice regions*/
			x_inside= x> inv_stats.at<int>(j,cv::CC_STAT_LEFT) and x < (inv_stats.at<int>(j,cv::CC_STAT_LEFT) + inv_stats.at<int>(j,cv::CC_STAT_WIDTH));
			y_inside= y> inv_stats.at<int>(j,cv::CC_STAT_TOP) and y < (inv_stats.at<int>(j,cv::CC_STAT_TOP) + inv_stats.at<int>(j,cv::CC_STAT_HEIGHT));
			if(x_inside and y_inside){
				int circle_area=stats.at<int>(i,cv::CC_STAT_AREA);
				/*On testing, most dots had area in [400,950] pixels after thresholding. This might be needed to be changed for different images*/
				if(circle_area>min_pixels and circle_area<max_pixels){
					counts_of_dice[j]++;
				}	
			}  
		}
	}
}
/*A function for wrting the number as well the sum of numbers on the image*/
void write_counts(int inv_number_of_labels, int counts_of_dice[], cv::Mat img, cv::Mat inv_stats){
	int sum=0;
	for (int i=1;i<inv_number_of_labels;i++){
		if (counts_of_dice[i]!=0){
			std::string text=std::to_string(counts_of_dice[i]);
			int x=inv_stats.at<int>(i,cv::CC_STAT_LEFT)+inv_stats.at<int>(i,cv::CC_STAT_WIDTH)+20;
			int y=inv_stats.at<int>(i,cv::CC_STAT_TOP)+0.5*inv_stats.at<int>(i,cv::CC_STAT_HEIGHT);
			cv::putText(img,text, cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,200,0),3);
			sum+=counts_of_dice[i];
		}
	}
	cv::putText(img,"Sum: "+std::to_string(sum), cv::Point(20,100), cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,200,0),3);	
}

/*A function to make images have a green border along boundaries similar to the example output*/ 
void create_borders(cv::Mat img, cv::Mat border){
	for (int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if (border.at<uchar>(i,j)==0){
				cv::Vec3b & color = img.at<cv::Vec3b>(i,j);
				color[1]=200;
				color[0]=0;
				color[2]=0;
			}
		}
	}
}

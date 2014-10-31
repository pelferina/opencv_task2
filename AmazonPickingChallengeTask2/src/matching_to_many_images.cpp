#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

string applName;
const string source_window = "Source Image";
const string result_window = "Result Image";


static void createWindows()
{
	/// Create windows
	namedWindow( source_window, CV_WINDOW_NORMAL );
	namedWindow( result_window, CV_WINDOW_NORMAL );
}

static void printPrompt( const string& applName )
{
    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [part1TextFileDir] [part2TextFileDir] [dirToSaveFinalImages]" << endl;
    cout << endl;
    exit(1);
}

/* Finds Region of interest within image
 * @param x,y coordinates of the leftmost corner of the desired rectangular region
 * @return roi Rectangular regions of interest
 */
Rect findroi(int x, int y)
{
	// Setup a Region Of Interest
	cv::Rect roi;
	roi.x = x;
	roi.y = y;
	roi.width = 2160;
	roi.height = 1216;

	return roi;
}

/*
 *  Transforms the corners of the chessboard
 * to be consistent with the resulting image
 */
void perspectiveTransformation(Mat src, vector<Point> corners, Mat &quad)
{

	// Define the corners of the destination image
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
	quad_pts.push_back(cv::Point2f(0, quad.rows));

	//Define the center points of the chessboards for transformation
	vector<Point2f>points;
	points.push_back(corners[0]);
	points.push_back(corners[1]);
	points.push_back(corners[2]);
	points.push_back(corners[3]);

	// Get transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(points, quad_pts);

	// Apply perspective transformation
	cv::warpPerspective(src, quad, transmtx, quad.size());
}

/* Finds Chess board Patterns
 * @img Corner Rectangles of the src
 * @param corners Centers of the chessboard patterns
 * @param result Index of the corner
 */
static void findChessboardPatterns(Mat &img, vector<Point> &corners, int result)
{
	Mat result1, result2, result3, result4;

	//Board width and height to be found
	int board_w = 3;
	int board_h = 3;

	bool found = false;

	vector<Point2f> ptvec;

	Size boardSize = Size( board_w, board_h );

	//Find the chess board pattern
	found = findChessboardCorners( img, boardSize, ptvec, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

	if (found)
	{
	  Mat viewGray;
	  cvtColor(img, viewGray, COLOR_BGR2GRAY);
	  cornerSubPix( viewGray, ptvec, Size(11,11),
	  Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.1 ));

	  //Adding the approximate centers to the corners vector
	  if(result == 2)
		  ptvec[4].x += 2160;
	  else if(result == 3)
		  ptvec[4].y += 1216;
	  else if(result == 4)
	  {
		  ptvec[4].x += 2160;
		  ptvec[4].y += 1216;
	  }
	  corners.push_back(ptvec[4]);
	}
}

static void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
{
    mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
            mask[i] = 1;
    }
}

static bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
                                      Ptr<FeatureDetector>& featureDetector,
                                      Ptr<DescriptorExtractor>& descriptorExtractor,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
    featureDetector = FeatureDetector::create( detectorType );
    descriptorExtractor = DescriptorExtractor::create( descriptorType );
    descriptorMatcher = DescriptorMatcher::create( matcherType );
    cout << ">" << endl;

    bool isCreated = !( featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty() );
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

    return isCreated;
}

static void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                      Ptr<FeatureDetector>& featureDetector )
{
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );
    cout << endl << "< Extracting keypoints from images..." << endl;
    detector.detect( queryImage, queryKeypoints );
    detector.detect( trainImages, trainKeypoints );
    cout << ">" << endl;
}

static void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
                         const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                         Ptr<DescriptorExtractor>& descriptorExtractor )
{
    cout << "< Computing descriptors for keypoints..." << endl;
    SurfDescriptorExtractor extractor;

    extractor.compute( queryImage, queryKeypoints, queryDescriptors );
    descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

    int totalTrainDesc = 0;
    for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
        totalTrainDesc += tdIter->rows;

    cout << "Query descriptors count: " << queryDescriptors.rows << "; Total train descriptors count: " << totalTrainDesc << endl;
    cout << ">" << endl;
}

static void matchDescriptors( const Mat& queryDescriptors, const vector<Mat>& trainDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    TickMeter tm;

    tm.start();
    descriptorMatcher->add( trainDescriptors );
    descriptorMatcher->train();
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();
    descriptorMatcher->match( queryDescriptors, matches );
    tm.stop();
    double matchTime = tm.getTimeMilli();

    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

    cout << "Number of matches: " << matches.size() << endl;
    cout << "Build time: " << buildTime << " ms; Match time: " << matchTime << " ms" << endl;
    cout << ">" << endl;
}

static void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
                       const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir )
{
    cout << "< Save results..." << endl;
    Mat drawImg;
    vector<char> mask;
    for( size_t i = 0; i < trainImages.size(); i++ )
    {
        if( !trainImages[i].empty() )
        {
            maskMatchesByTrainImgIdx( matches, (int)i, mask );
            drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
                         matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask );
            string filename = resultDir + "/res_" + trainImagesNames[i];
            if( !imwrite( filename, drawImg ) )
                cout << "Image " << filename << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;
        }
    }
    cout << ">" << endl;
}

static void readImages( const string fileDir, string& imageDirName, vector<string>& imageFileNames )
{
    imageFileNames.clear();

    ifstream file( fileDir.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = fileDir.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = fileDir.rfind('/');
        dlmtr = '/';
    }
    imageDirName = pos == string::npos ? "" : fileDir.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        imageFileNames.push_back(str);
    }
    file.close();
}

static bool readImagesFromFile(const string& textFileDir, vector <Mat>& images, vector<string> &imageFileNames, int flag)
{
	string imageDirName;
	cout<<"<"<<endl;
	readImages(textFileDir, imageDirName, imageFileNames);

	if( imageFileNames.empty() )
	{
		cout << "Train image filenames can not be read." << endl << ">" << endl;
		return false;
	}
	int readImageCount = 0;
	for( size_t i = 0; i < imageFileNames.size(); i++ )
	{
		string filename = imageDirName + imageFileNames[i];
		Mat img = imread( filename, flag );
		if(CV_LOAD_IMAGE_GRAYSCALE == flag)
		{
			Size t((int)(img.cols*30)/100, (int)(img.rows*30)/100);
			resize(img, img, t);
		}
		if( img.empty() )
			cout << filename << " can not be read." << endl;
		else
			readImageCount++;
		images.push_back( img );
	}
	if( !readImageCount )
	{
		cout << "All image(s) can not be read." << endl << ">" << endl;
		return false;
	}
	else
		cout << readImageCount << " image(s) were read." << endl;
	cout << ">" << endl;

	return true;
}

static void part1Execute(string applName, vector<Mat> images, vector<string> trainImagesNames, const string dirToSaveFinalImages)
{
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor = new BriefDescriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;

	if( !createDetectorDescriptorMatcher( "SURF", "SURF", "FlannBased", featureDetector, descriptorExtractor, descriptorMatcher ) )
	{
		printPrompt(applName);
	}

	vector<KeyPoint> queryKeypoints;
	vector<vector<KeyPoint> > trainKeypoints;

	Mat queryImage = images[0];
	vector<Mat> trainImages (images.begin() + 1, images.end());
	detectKeypoints( queryImage, queryKeypoints, trainImages, trainKeypoints, featureDetector );
	Mat queryDescriptors;
	vector<Mat> trainDescriptors;
	computeDescriptors( queryImage, queryKeypoints, queryDescriptors,
						trainImages, trainKeypoints, trainDescriptors,
						descriptorExtractor );

	vector<DMatch> matches;
	matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );

	saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
					  matches, trainImagesNames, dirToSaveFinalImages );
}

static void part2Execute(string applName, Mat src, string imageName, string location)
{
	Mat result2, result3, result4;
	vector<Point> corners;

	// Crop the original image to the area defined by ROI (Regions of Interest)
	// ROI = Four corners of the image for ease of finding chess board patterns within these areas
	result4 = src(findroi(2160, 1216));
	result2 = src(findroi(2160, 0));
	result3 = src(findroi(0, 1216));

	// Manually determine the first chessboard pattern's center as
	// it is unrecognizable in the image
	Point tl;
	tl.x = 1050;
	tl.y = 200;
	corners.push_back(tl);

	// Find the rest of the chessboard patterns
	// and approximate their centers
	findChessboardPatterns(result2, corners, 2);
	findChessboardPatterns(result4, corners, 4);
	findChessboardPatterns(result3, corners, 3);

	createWindows();

	//Show Original Image
	imshow( source_window, src);

	// Define the destination image as required
	Mat quad = cv::Mat::zeros(540, 960, CV_32FC2);

	//Transform the perspective with the given corners
	perspectiveTransformation(src, corners, quad);

	string filename = location + "/" + imageName;
	if( !imwrite( filename, quad ) )
		cout << "Image " << filename << " can not be saved (may be because directory " << location << " does not exist)." << endl;
	//Show Transformed Image
	imshow(result_window, quad);
	waitKey(0);
}

int main(int argc, char** argv)
{
	applName = argv[0];
    if( argc <= 2)
    {
       printPrompt(applName);
    }

    const string part1TextFile = argv[1];
    const string part2TextFile = argv[2];
    const string dirToSaveFinalImages = argv[3];

    vector<Mat> part1Images, part2Images;
	vector<string> imageFileNamesPart1, imageFileNamesPart2;

    if(readImagesFromFile(part1TextFile, part1Images, imageFileNamesPart1, CV_LOAD_IMAGE_GRAYSCALE) &&
    		readImagesFromFile(part2TextFile, part2Images, imageFileNamesPart2, IMREAD_COLOR))
    {
    	part1Execute(applName, part1Images, imageFileNamesPart1, dirToSaveFinalImages);
    	part2Execute(applName, part2Images[0], imageFileNamesPart2[0], dirToSaveFinalImages);	//since there is only one image for part 2
    }
}

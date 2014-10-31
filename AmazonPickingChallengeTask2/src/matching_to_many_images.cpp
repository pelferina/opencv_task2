#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

string applName;
const string source_window = "Source Image";

static void createWindows()
{
	/// Create windows
	namedWindow( source_window, CV_WINDOW_NORMAL );
}

static void printPrompt( const string& applName )
{
    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [part1TextFileDir] [part2TextFileDir] [dirToSaveFinalImages]" << endl;
    cout << endl;
    exit(1);
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
    }
}

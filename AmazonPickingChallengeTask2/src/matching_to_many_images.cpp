#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

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

static bool readImagesFromFile(const string& textFileDir, vector <Mat>& images, int flag)
{
	string imageDirName;
	vector<string> imageFileNames;
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

int main(int argc, char** argv)
{
	const string applName = argv[0];
    if( argc <= 2)
    {
       printPrompt(applName);
    }

    const string part1TextFile = argv[1];
    const string part2TextFile = argv[2];
    const string dirToSaveFinalImages = argv[3];

    vector<Mat> part1Images, part2Images;

    if(readImagesFromFile(part1TextFile, part1Images, CV_LOAD_IMAGE_GRAYSCALE) &&
    		readImagesFromFile(part2TextFile, part2Images, IMREAD_COLOR))
    {
    	createWindows();
    	for(int i = 0; i < part1Images.size(); i++)
    	{
    		imshow(source_window, part1Images[i]);
    		waitKey(1000);
    	}
    	for(int i = 0; i < part2Images.size(); i++)
		{
			imshow(source_window, part2Images[i]);
			waitKey(1000);
		}

    }
}

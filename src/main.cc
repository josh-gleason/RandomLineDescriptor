#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>

using namespace cv;
using namespace std;

typedef pair<Point2d,Point2d> Line;

const int CV_FTYPE = CV_64FC1;
typedef double FType;

struct ProgramSettings {
   struct MserSettings {
      int delta;
      double minArea;
      double maxArea;
      double maxVariation;
      double minDiversity;
   };
   
   MserSettings mser;
};

void getEllipsePoints(vector<Point2d>& vertices, const RotatedRect& box, size_t points=1000U);
void drawEllipse(Mat& image, const RotatedRect& box, const Scalar& color = Scalar(0,0,255));
void drawEllipse(Mat& image, const vector<Point2d>& vertices, const Scalar& color = Scalar(0,0,255));
void genLines(vector<Line>& pairs, const vector<Point2d>& vertices, double minDist=1.0, size_t pairCount=50);

void genLines(vector<Line>& lines, const vector<Point2d>& vertices, double minDist, size_t pairCount)
{
   lines.resize(pairCount);

   for ( int i = 0; i < pairCount; ++i )
   {
      Line p;

      int count = 0;
      do {
         p.first = vertices[rand() % vertices.size()];
         p.second = vertices[rand() % vertices.size()];
         count++;
      } while (sqrt((p.first.x-p.second.x)*(p.first.x-p.second.x)+
                    (p.first.y-p.second.y)*(p.first.y-p.second.y)) < minDist && count < 1000);

      // tried too many times, couldn't find a valid pair
      if ( count == 1000 )
      {
         cout << "Error: " << __LINE__ << ':' << __FILE__ << endl;
         return;
      }

      lines[i] = p;
   }
}

void getEllipsePoints(vector<Point2d>& vertices, const RotatedRect& box, size_t points)
{
   RotatedRect r = box;
  
   if ( r.size.height < r.size.width )
   {
      r.size.height = box.size.width;
      r.size.width = box.size.height;

      r.angle = (box.angle + 90.0);
   }
   
   r.angle *= M_PI / 180.0f;

   double a = r.size.height / 2.0;
   double b = r.size.width / 2.0;
   
   double step = 2 * M_PI / points;
   double theta = 0.0;

   double bCos, aSin, radius;

   vertices.resize(points);
   for ( size_t i = 0; i < points; ++i )
   {
      bCos = a * cos(theta);
      aSin = b * sin(theta);
      radius = a * b / sqrt(bCos * bCos + aSin * aSin);

      vertices[i] = Point2f(
         r.center.x + radius * cos(theta+r.angle),
         r.center.y + radius * sin(theta+r.angle));

      theta += step;
   }

}

void drawEllipse(Mat& image, const vector<Point2d>& vertices, const Scalar& color)
{
   for ( const Point2d& p : vertices )
      image.at<Vec3b>(p) = Vec3b(color[0],color[1],color[2]);
}

void drawEllipse(Mat& image, const RotatedRect& box, const Scalar& color)
{
   vector<Point2d> vertices(1000U);
   getEllipsePoints(vertices, box, 1000U);

   drawEllipse(image, vertices, color);

/*
   ellipse(image, box, Scalar(255,0,0));

   Point2f verts[4];
   box.points(verts);
   for ( int i = 0; i < 4; ++i )
      line(image, verts[i], verts[(i+1)%4], Scalar(255,0,0));
*/
}

void drawLines(Mat& image, const vector<Line>& lines, const Scalar& color)
{
   for ( const Line& p : lines )
   {
      line(image, p.first, p.second, color);
   }
}

double distFunc(const Mat& first, const Mat& second)
{
   Mat diff = first - second;

   // square
   diff = diff.mul(diff);

   // set last element (holds the mean before normalization) to zero
   diff.at<FType>(diff.total()-1) = 0;
  
   // mean is pre-subtracted
   return sum(diff)[0];
}

void buildFeatures(vector<Mat>& features, const Mat& image, vector<Line>& lines, int featureSize)
{
   features.clear();
   features.resize(lines.size());

   // features
   // pixel1, pixel2, pixel3 ..., pixelN, mean

   for ( size_t i = 0; i < lines.size(); ++i )
   {
      Mat& feat = features[i];
      feat.create(1, featureSize, CV_FTYPE);

      Point2d& p1 = lines[i].first;
      Point2d& p2 = lines[i].second;

      Point2d sample = p1;

      Point2d step((p2.x - p1.x) / (featureSize - 1.0),
                   (p2.y - p1.y) / (featureSize - 1.0));

      double sum = 0.0;
      
      for ( int j = 0; j < featureSize-1; ++j )
      {
         feat.at<FType>(j) = image.at<unsigned char>(Point2i(sample.x,sample.y));
         sum += feat.at<FType>(j);

         // increment step
         sample.x += step.x;
         sample.y += step.y;
      }

      // store mean
      FType mean = (FType)(sum / (featureSize - 1.0));
      
      // subtract mean for feat
      feat -= mean;
     
      // save the old mean so we can get original value back
      feat.at<FType>(featureSize-1) = mean;
   }
}

int main(int argc, char *argv[])
{
   // ideas
   // Blur region based on its size before sampling
   // Reject threshold based on how many matches belonged to first vs. second class
   //    Reject threhold for precision recall?

   // image of black
   Mat img = imread(argv[1]);
   Mat grayImg;
   cvtColor(img, grayImg, CV_RGB2GRAY);

   RotatedRect r(Point2f(250.0f, 250.0f), Size2f(400.0f,300.0f), 100);

   double threshold = min(r.size.width, r.size.height)/2.0;

   vector<Point2d> vertices(1000U);
   vector<Line> lines;
   vector<Mat> features;
   
   double featureSize = 21; // 20 samples and 1 mean stored

   getEllipsePoints(vertices, r, 1000U);
   genLines(lines, vertices, threshold, 50);

   drawEllipse(img, vertices, Scalar(255,0,0));
   drawLines(img, lines, Scalar(255,255,255));
   
   buildFeatures(features, grayImg, lines, featureSize);


   Mat a(1,6,CV_FTYPE);
   Mat b(1,6,CV_FTYPE);
   
   a.at<FType>(0) = -20;
   a.at<FType>(1) = -10;
   a.at<FType>(2) = 0;
   a.at<FType>(3) = 10;
   a.at<FType>(4) = 20;
   a.at<FType>(5) = 20;

   b.at<FType>(0) = -40;
   b.at<FType>(1) = -20;
   b.at<FType>(2) = 0;
   b.at<FType>(3) = 20;
   b.at<FType>(4) = 40;
   b.at<FType>(5) = 40;
   
   Mat c = a.clone();
   Mat d = b.clone();

   c += 20;
   d += 40;

   c.at<FType>(5) = 20;
   d.at<FType>(5) = 40;

   cout << distFunc(a, b) << endl;
   cout << distFunc(c, d) << endl;
   cout << distFunc(features[1], features[0]) << endl;

   imshow("Image", img);
   waitKey(0);

   return 0;

#if 0
   if ( argc < 3 ) {
      cout << "Usage: " << argv[0] << " <input image> <output image>" << endl;
      return -1;
   }

   // TODO Load settings from file
   ProgramSettings settings;

   settings.mser.delta = 2;
   settings.mser.minArea = 0.00015;
   settings.mser.maxArea = 0.35;
   settings.mser.maxVariation = 0.25;
   settings.mser.minDiversity = 0.2;

   // images
   Mat image = imread(argv[1]);

   // convert to gray for MSER
   Mat grayImage;
   cvtColor(image, grayImage, CV_RGB2GRAY);

   // MSER from Dubout
   // delta = 2
   // minArea = 0.0001*imageArea
   // maxArea = 0.5*imageArea
   // maxVariation = 0.4
   // minDiversity = 0.33

   // MSER from OpenCV
   // delta = 5
   // minArea = 60
   // maxArea = 14400
   // maxVariation = 0.25
   // minDiversity = 0.2
   //    maxEvolution = 200
   //    areaThreshold=1.01
   //    minMargin = 0.003
   //    edgeBlurSize = 5

   // Construct an MSER detector
   MserFeatureDetector mserDetector(
      settings.mser.delta,
      settings.mser.minArea * image.size().area(),
      settings.mser.maxArea * image.size().area(),
      settings.mser.maxVariation,
      settings.mser.minDiversity);

   vector<vector<Point>> keypoints;

   mserDetector(grayImage, keypoints);

   for ( const vector<Point>& contour : keypoints ) {
      RotatedRect box = minAreaRect(contour);
//      ellipse( image, box, Scalar(255, 0, 0), 2 );
      drawEllipse( image, box, Scalar(255, 0, 0) );
   }

   imwrite(argv[2], image);

   waitKey(0);

   return 0;
#endif
}

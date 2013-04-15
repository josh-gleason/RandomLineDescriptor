#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include <limits>

using namespace cv;
using namespace std;

//#define DRAW_CIRCLE

typedef pair<Point2d,Point2d> Line;

const int CV_FTYPE = CV_32FC1;
typedef float FType;

struct ProgramSettings {
   struct MserSettings {
      MserSettings() :
         delta(2),
         minArea(0.00015),
         maxArea(0.35),
         maxVariation(0.25),
         minDiversity(0.2)
      {}

      int delta;
      double minArea;
      double maxArea;
      double maxVariation;
      double minDiversity;
   };
   
   struct DescriptorSettings {
      DescriptorSettings() :
         ellipseSize(2.0),
         ellipsePoints(512U),
         l(32),
         Nk(150),
         N(150),
         minDist(0.25),
         k1(1.0),
         p1(1.0),
         p2(1.0),
         w1(1.0),
         w2(1.0),
         smoothing(4.0)
      {}

      double ellipseSize;  // scale factor of ellipse to min bounding box
      size_t ellipsePoints;   // number of points to sample around perimeter of ellipse

      int l;   // number of points per line
      int Nk;  // number of lines per region in reference region
      int N;   // number of lines per region in test region

      double minDist; // minimum allowed length of line (with respect to sqrt(area) of region)

      double k1;  // Dmax weighting (0,1]
      double p1;  // controls shape of confidence function
      double p2;  // controls shape of confidence function
      double w1;  // weight for distance
      double w2;  // weight for error

      double smoothing; // strength of smoothing (relative to scale of region)
   };

   MserSettings mser;
   
   DescriptorSettings descriptor;
};

struct Region {
   RotatedRect ellipse;
   vector<Line> lines;
   vector<Mat> descriptors;
   vector<double> mean;
   vector<double> err;

   long int baseIdx;

   double errMax;
};

void getEllipsePoints(vector<Point2d>& vertices, const RotatedRect& box, size_t points=1000U);
void drawEllipse(Mat& image, const RotatedRect& box, const Scalar& color = Scalar(0,0,255));
void drawEllipse(Mat& image, const vector<Point2d>& vertices, const Scalar& color = Scalar(0,0,255));
void genLines(Region& region, const vector<Point2d>& vertices, double minDist=1.0, size_t pairCount=50);

void genLines(Region& region, const vector<Point2d>& vertices, double minDist, size_t pairCount)
{
   region.lines.resize(pairCount);

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

      region.lines[i] = p;
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
   // difference
   Mat diff = first - second;
   
   // squared
   diff = diff.mul(diff);
  
   // mean is pre-subtracted so no -delta function
   return sum(diff)[0];
}

// image must be grayscale
void buildFeatures(Region& region, const Mat& image, size_t lineCount,
   int featureSize, double smoothing, double minDist)
{
   // compute affine transform for normalization
   Point2f src[4];
   region.ellipse.points(src);

   int ksize = smoothing * 4;
   if ( ksize % 2 == 0 ) ksize++;

   const int imageSize = 255;
   const int border = ksize/2;

   minDist *= imageSize;

   Point2f dst[3] = {Point2f(border, imageSize - border),
                     Point2f(border, border),
                     Point2f(imageSize - border, border)};

   Mat transform = getAffineTransform(src, dst);

   // transform image
   Mat normImg(imageSize, imageSize, image.type());
   warpAffine(image, normImg, transform, Size(imageSize, imageSize),
              INTER_LINEAR, BORDER_REFLECT);

   // smoooth image
   if ( smoothing > 0.0 )
      GaussianBlur(normImg, normImg, Size(ksize,ksize), smoothing);

   Point center(imageSize/2, imageSize/2);
   double radius = imageSize/2 - border;

#ifdef DRAW_CIRCLE
   // draw circle
   circle(normImg, center, radius, Scalar(255,0,0)); 

   Mat imageColor;
   cvtColor(image, imageColor, CV_GRAY2RGB);

   drawEllipse(imageColor, region.ellipse, Scalar(255,255,0));

   imshow("ImageRegion", normImg);
   imshow("Image", imageColor);
   waitKey(0);
#endif

   // compute descriptor
   region.descriptors.resize(lineCount);
   region.mean.resize(lineCount);
   region.err.resize(lineCount);
   region.lines.resize(lineCount);

   // features
   // pixel1, pixel2, pixel3 ..., pixelN, mean, err

   // mean = sum(pixels)/(featureSize-2)
   // err = sum(|pixel_n - pixel_n+1|)_n={1,featureSize-3}/(featureSize-3)

   for ( size_t i = 0; i < lineCount; ++i )
   {
      // generate random line
      double theta1, theta2;

      // generate random point
      Point2d p1, p2;
      do {
         theta1 = ((rand() % 10000) / 10000.0) * 2 * M_PI;
         theta2 = ((rand() % 10000) / 10000.0) * 2 * M_PI;

         p1.x = center.x + radius*cos(theta1);
         p1.y = center.y + radius*sin(theta1);
         p2.x = center.x + radius*cos(theta2);
         p2.y = center.y + radius*sin(theta2);
      } while (sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)) < minDist);

      // save the lines in case they are needed later
      region.lines[i] = Line(p1,p2);

      Mat& feat = region.descriptors[i];
      feat.create(1, featureSize, CV_FTYPE);

      Point2d sample = p1;
      Point2d step((p2.x - p1.x) / (featureSize - 1.0),
                   (p2.y - p1.y) / (featureSize - 1.0));

      double sum = 0.0;
      double diffSum = 0.0;
      
      for ( int j = 0; j < featureSize; ++j )
      {
         feat.at<FType>(j) = normImg.at<unsigned char>(Point2i(sample.x,sample.y));
         sum += feat.at<FType>(j);

         if ( j != 0 )
            diffSum += fabs(feat.at<FType>(j-1)-feat.at<FType>(j));

         // increment step
         sample.x += step.x;
         sample.y += step.y;
      }

      // store mean
      region.mean[i] = (FType)(sum / (featureSize));
      region.err[i] = (FType)(diffSum / (featureSize - 1.0));

      // subtract mean for feat
      feat -= region.mean[i];
   }

   region.errMax = *max_element(region.err.begin(), region.err.end());
}

// color image input
void extractRegions(vector<Region>& regions, const Mat& image, const ProgramSettings& settings, bool referenceImage)
{
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

   size_t regionCount = 0;
   vector<RotatedRect> ellipses;
   for ( const vector<Point>& contour : keypoints ) {
      RotatedRect box = minAreaRect(contour);
      box.size.width *= settings.descriptor.ellipseSize;
      box.size.height *= settings.descriptor.ellipseSize;
      // TODO reject very high eccentricity regions
      if ( (Rect(Point(0,0), image.size()) & box.boundingRect()).size() ==
            box.boundingRect().size() )
      {
//         drawEllipse( image, box, Scalar(255, 0, 0) );
         ellipses.push_back(box);
      }
   }

   ///////// Build descriptors
   // initialize region list
   regions.resize(ellipses.size());

   const ProgramSettings::DescriptorSettings& s = settings.descriptor;
   
   long int baseIdx = 0;
   for ( int i = 0; i < ellipses.size(); ++i )
   {
      RotatedRect& r = ellipses[i];
      regions[i].ellipse = r;
      if ( referenceImage ) {
         buildFeatures(regions[i], grayImage, s.Nk, s.l, s.smoothing, s.minDist);
         regions[i].baseIdx = baseIdx;
         baseIdx += s.Nk;
      } else {
         buildFeatures(regions[i], grayImage, s.N, s.l, s.smoothing, s.minDist);
      }
   }
}

void findMatches(vector<Line>& matches, const vector<Region>& refRegions,
   const vector<Region>& matchRegions)
{
   Mat refDescriptors(refRegions.size()*refRegions[0].descriptors.size(),
      refRegions[0].descriptors[0].cols, CV_FTYPE);
   
   Mat matchDescriptors(matchRegions.size()*matchRegions[0].descriptors.size(),
      matchRegions[0].descriptors[0].cols, CV_FTYPE);

   size_t Nk = refRegions[0].descriptors.size();
   size_t N = matchRegions[0].descriptors.size();

   for ( size_t i = 0; i < refRegions.size(); ++i )
      for ( size_t j = 0; j < Nk; ++j )
         for ( size_t k = 0; k < refRegions[i].descriptors[j].size().width; ++k )
            refDescriptors.at<FType>(i*N+j,k) =
               refRegions[i].descriptors[j].at<FType>(0,k);

   for ( size_t i = 0; i < matchRegions.size(); ++i )
      for ( size_t j = 0; j < N; ++j )
         for ( size_t k = 0; k < matchRegions[i].descriptors[j].size().width; ++k )
            matchDescriptors.at<FType>(i*N+j,k) =
               matchRegions[i].descriptors[j].at<FType>(0,k);

   FlannBasedMatcher matcher;
   std::vector<DMatch> matchList;
   matcher.match(matchDescriptors, refDescriptors, matchList);

   int count = 0;
   for ( size_t i = 0; i < matchRegions.size(); ++i )
   {
      const Region& matchReg = matchRegions[i];

      // will hold closest matches
      vector<long int> matchIdx(N);
      vector<double> matchDist(N);

      for ( size_t j = 0; j < N; ++j )
      {
         const Mat& d = matchReg.descriptors[i];
         double mean = matchReg.mean[i];
         double err = matchReg.err[i];
         double errMax = matchReg.errMax;
     
         matchIdx[j] = matchList[i*N+j].trainIdx;
         matchDist[j] = matchList[i*N+j].distance * matchList[i*N+j].distance;
      }

      // now compute region matches and confidence
   
   }


}

int main(int argc, char *argv[])
{
   // ideas
   // Blur region based on its size before sampling
   // Reject threshold based on how many matches belonged to first vs. second class
   //    Reject threhold for precision recall?

   srand(time(0));
   
//   if ( argc < 4 ) {
   if ( argc < 1 ) {
      cout << "Usage: " << argv[0] << " <input image> <match image> <output image>" << endl;
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
   Mat refImage = imread(argv[1]);
   Mat matchImage = imread(argv[1]);

   vector<Region> refRegions, matchRegions;
   
   extractRegions(refRegions, refImage, settings, true);
   extractRegions(matchRegions, matchImage, settings, false);

   vector<Line> matches;

   // find matching regions
   findMatches(matches, refRegions, matchRegions);

   cout << matches.size() << endl;

   waitKey(0);

   return 0;
}


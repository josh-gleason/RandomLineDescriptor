#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include <limits>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <deque>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "settings.h"
#include "results.h"

using namespace cv;
using namespace std;

namespace bg = boost::geometry;

//#define CUBIC_INTERPOLATION

// ideas
// Reject threshold based on how many matches belonged to first vs. second class
//    Reject threhold for precision recall?

typedef pair<Point2d,Point2d> Line;
//typedef pair<Line, double> Match; // matching region and CMF

// TODO save match indices
struct Match {
   Match(const Point2d& p1, const Point2d& p2, int refIdx, int matIdx, double c, double d, int m)
      : refIndex(refIdx), matchIndex(matIdx), matchedLines(m), dist(d), cmf(c)
   {
      points[0] = p1;
      points[1] = p2;
   }

   // first point on match image
   // second point on reference image
   Point2d points[2];

   int refIndex;
   int matchIndex;

   int matchedLines;
   double dist;
   double cmf;
};

const int CV_FTYPE = CV_32FC1;
typedef float FType;

struct Region {
   RotatedRect ellipse;
   vector<Line> lines;
   vector<Mat> descriptors;
   vector<double> mean;
   vector<double> err;

   long int baseIdx;

   double meanErr;
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
   double theta = 2 * M_PI;

   double bCos, aSin, radius;

   // clockwise to make area work with boost
   vertices.resize(points);
   for ( size_t i = 0; i < points; ++i )
   {
      bCos = a * cos(theta);
      aSin = b * sin(theta);
      radius = a * b / sqrt(bCos * bCos + aSin * aSin);

      vertices[i] = Point2f(
         r.center.x + radius * cos(theta+r.angle),
         r.center.y + radius * sin(theta+r.angle));

      theta -= step;
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

double dist(const Point2d& p1, const Point2d& p2)
{
   return sqrt( (p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y) );
}

void buildFeatures(Region& region, const Mat& image, size_t lineCount, const ProgramSettings& settings)
{
   static vector<Point2d> verts;
   getEllipsePoints(verts, region.ellipse, settings.descriptor.ellipsePoints);
   
   // resize vectors
   region.descriptors.resize(lineCount);
   region.mean.resize(lineCount);
   region.err.resize(lineCount);
   region.lines.resize(lineCount);

   // compute descriptors
   for ( int idx = 0; idx < lineCount; ++idx ) {
      // choose two random vertices
      int idx1, idx2;

      // pick random points
      if ( settings.descriptor.minDist <= 0 ) {
         do {
            idx1 = rand() % settings.descriptor.ellipsePoints;
            idx2 = rand() % settings.descriptor.ellipsePoints;
         } while ( idx1 == idx2 );
      } else {
         do {
            idx1 = rand() % settings.descriptor.ellipsePoints;
            idx2 = rand() % settings.descriptor.ellipsePoints;
         } while ( dist(verts[idx1],verts[idx2]) <= settings.descriptor.minDist );
      }

      // create line iterator
      LineIterator line(image, verts[idx1], verts[idx2]);

      int lineLen = line.count;   // number of pixels in line

      // copy line to vector for iteration
      Mat lineMat(1, lineLen, CV_FTYPE);
      for ( int i = 0; i < line.count; ++i, ++line )
         lineMat.at<FType>(0,i) = (FType)(**line);

      // create descriptor
      region.descriptors[idx].create(Size(settings.descriptor.l, 1), CV_FTYPE);

      // set first value
      region.descriptors[idx].at<FType>(0,0) = lineMat.at<FType>(0,0);
      
      double sum = lineMat.at<FType>(0,0);
      double diffs = 0;

      // linear interpolate
      for ( int i = 1; i < settings.descriptor.l; ++i )
      {
         double x = (double)i * (lineLen-1.0) / (settings.descriptor.l-1.0);
         if ( fmod(x,1.0) == 0.0 ) {
            region.descriptors[idx].at<FType>(0,i) = lineMat.at<FType>(0,(int)x);
         } else {
            double x1Val = lineMat.at<FType>(0,(int)x);
            double x2Val = lineMat.at<FType>(0,((int)x)+1);
          
            x = fmod(x,1.0);   // floating point part of x

            region.descriptors[idx].at<FType>(0,i) = x1Val + x * (x2Val - x1Val);
         }

         diffs += fabs(region.descriptors[idx].at<FType>(0,i) - region.descriptors[idx].at<FType>(0,i-1));
         sum += region.descriptors[idx].at<FType>(0,i);
      }

      region.err[idx] = diffs / (FType)(settings.descriptor.l - 1);
      region.mean[idx] = sum / (FType)settings.descriptor.l;

      region.meanErr += region.err[idx];

      region.descriptors[idx] -= region.mean[idx];
   }
  
   region.meanErr /= (double)lineCount;
}

// image must be grayscale
void buildSmoothedFeatures(Region& region, const Mat& image, size_t lineCount,
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
#ifdef CUBIC_INTERPOLATION
              INTER_CUBIC, BORDER_REFLECT);
#else
              INTER_LINEAR, BORDER_REFLECT);
#endif
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

#ifdef DRAW_CIRCLE
      // draw line
      line(normImg, p1, p2, Scalar(0,0,0), 1);
#endif
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
   
#ifdef DRAW_CIRCLE
   imshow("ImageRegion", normImg);
   waitKey(0);
#endif

//   region.errMax = *max_element(region.err.begin(), region.err.end());
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
      settings.mser.minArea * (settings.mser.useRelativeArea ? image.size().area() : 1.0),
      settings.mser.maxArea * (settings.mser.useRelativeArea ? image.size().area() : 1.0),
      settings.mser.maxVariation,
      settings.mser.minDiversity);

   vector<vector<Point>> keypoints;

   mserDetector(grayImage, keypoints);

   // shuffle keypoints
   random_shuffle(keypoints.begin(),keypoints.end());

   size_t regionCount = 0;
   vector<RotatedRect> ellipses;
   for ( const vector<Point>& contour : keypoints ) {
      RotatedRect box = minAreaRect(contour);
      box.size.width *= settings.descriptor.ellipseSize;
      box.size.height *= settings.descriptor.ellipseSize;

      // TODO reject very high eccentricity regions
      if ((Rect(Point(0,0), image.size()) & box.boundingRect()).size()
         == box.boundingRect().size())
      {
         ellipses.push_back(box);
         regionCount++;
      }
      if (regionCount > settings.mser.maxRegions)
         break;;
   }

   ///////// Build descriptors
   // initialize region list
   regions.resize(ellipses.size());

   const ProgramSettings::DescriptorSettings& s = settings.descriptor;
   
   vector<int> badRegions;
   for ( int i = 0; i < ellipses.size(); ++i )
   {
      RotatedRect& r = ellipses[i];
      regions[i].ellipse = r;

      if ( settings.descriptor.smoothRegion ) {  // SLOW VERSION
         if ( referenceImage ) {
            buildSmoothedFeatures(regions[i], grayImage, s.Nk, s.l, s.smoothing, s.minDist); 
         } else {
            buildSmoothedFeatures(regions[i], grayImage, s.N, s.l, s.smoothing, s.minDist); 
         }
      } else { // Much faster
         if ( referenceImage ) {
            buildFeatures(regions[i], grayImage, s.Nk, settings); 
         } else {
            buildFeatures(regions[i], grayImage, s.N, settings); 
         }
      }

      if ( regions[i].meanErr < settings.descriptor.minMeanErr )
      {
         // reject
         badRegions.push_back(i);
      }
   }

   cout << "Removed " << badRegions.size() << " Regions" << endl;

   for ( int i = 0; i < badRegions.size(); ++i )
      regions.erase(regions.begin() + (badRegions[i] - i));

   cout << "Detected " << regions.size() << " Regions" << endl;
}

void findMatches(vector<Match>& matches, const vector<Region>& refRegions,
   const vector<Region>& matchRegions, const ProgramSettings& settings)
{
   size_t Nk = settings.descriptor.Nk;
   size_t N = settings.descriptor.N;
   size_t l = settings.descriptor.l;

   // initialize matrices
   Mat refDescriptors(refRegions.size()*Nk, l, CV_FTYPE);
   Mat matchDescriptors(matchRegions.size()*N, l, CV_FTYPE);

   // copy descriptors over
   for ( size_t i = 0; i < refRegions.size(); ++i )
      for ( size_t j = 0; j < Nk; ++j )
         refRegions[i].descriptors[j].copyTo(
            refDescriptors.row(i*Nk+j));
   
   for ( size_t i = 0; i < matchRegions.size(); ++i )
      for ( size_t j = 0; j < N; ++j )
         matchRegions[i].descriptors[j].copyTo(
            matchDescriptors.row(i*N+j));
   
   // Create matcher and match
   FlannBasedMatcher matcher(new flann::KDTreeIndexParams(settings.descriptor.kdTrees));
   std::vector<DMatch> matchList;
   matcher.match(matchDescriptors, refDescriptors, matchList);

   for ( size_t i = 0; i < matchRegions.size(); ++i )
   {
      const Region& matchReg = matchRegions[i];

      // will hold closest matches
      vector<long int> matchIdx(N);
      vector<double> matchDist(N);
      vector<int> matchedRegion(N);
      vector<int> matchedDesc(N);
      vector<double> conf(N);
      vector<double> err(N);

      int matchRegion;
      int matchDesc;
      
      for ( size_t j = 0; j < N; ++j )
      {

         matchIdx[j] = matchList[i*N+j].trainIdx;
         matchDist[j] = matchList[i*N+j].distance;
         
         // set matchRegion and matchDesc
         matchRegion = matchIdx[j] / Nk;
         matchDesc = matchIdx[j] % Nk;
        
         matchedRegion[j] = matchRegion;
         matchedDesc[j] = matchDesc;
         err[j] = refRegions[matchedRegion[j]].err[matchedDesc[j]];
      }

      // now compute region matches and confidence
      double Dmax = matchDist[0];
      double Dmin = Dmax;

      double errMax = err[0]; 
      
      for ( size_t j = 1; j < N; ++j )
      {
         // get Dmax
         if ( matchDist[j] > Dmax )
            Dmax = matchDist[j];
         if ( matchDist[j] < Dmin )
            Dmin = matchDist[j];
         if ( err[j] > errMax )
            errMax = err[j]; 
      }

      // multiply by k1
      Dmax *= settings.descriptor.k1;
      
      double p1 = settings.descriptor.p1;
      double p2 = settings.descriptor.p2;
      double w1 = settings.descriptor.w1;
      double w2 = settings.descriptor.w2;

      // compute confidence
      for ( size_t j = 0; j < N; ++j )
      {
         double Dj = matchDist[j];
         double errj = err[j];
         if ( matchDist[j] > Dmax ) {
            conf[j] = 0;
         } else {
            conf[j] = pow(w1 * (Dmax - Dj)/(Dmax - Dmin), p1)*
                      pow(w2 * errj / errMax, p2);
         }
      }

      // TC holds the total confidence (sum) for all lines matched to
      // a class.  TCidx holds the indices of which classes TC is refering
      // to.
      vector<double> TC;
      vector<int> TCidx;
      vector<int> TCcount;

      // compute TC for matched found classes
      for ( size_t j = 0; j < N; ++j )
      {
         if ( count(TCidx.begin(), TCidx.end(), matchedRegion[j]) == 0 ) {
            TC.push_back(conf[j]);
            TCidx.push_back(matchedRegion[j]);
            TCcount.push_back(1);
         } else { // already exists
            size_t k = 0;
            while ( TCidx[k] != matchedRegion[j] ) {
               ++k;
            }
            TC[k] += conf[j];
            TCcount[k]++;
         }
      }
   
      // find max and second max TC
      double TCmax = TC[0];
      int TCmaxIdx = TCidx[0];
      double TCmax2 = TCmax;
      int TCmaxIdx2 = TCmaxIdx;
      int TCmaxCount = TCcount[0];

      for ( size_t j = 0; j < TCidx.size(); ++j )
      {
         if ( TC[j] > TCmax )
         {
            TCmax2 = TCmax;
            TCmaxIdx2 = TCmaxIdx2;
            TCmax = TC[j];
            TCmaxIdx = TCidx[j];
            
            TCmaxCount = TCcount[j];
         } else if ( TC[j] > TCmax2 ) {
            TCmax2 = TC[j];
            TCmaxIdx = TCidx[j];
         }
      }

      // determine matched region and if match has occured
      double cmf = (TCmax - TCmax2)/TCmax2;
      
      // spacial distance between top two matches
      Point2d pt1(refRegions[TCmaxIdx].ellipse.center.x, refRegions[TCmaxIdx].ellipse.center.y);
      Point2d pt2(refRegions[TCmaxIdx2].ellipse.center.y, refRegions[TCmaxIdx2].ellipse.center.y);
      double distance = dist(pt1,pt2); //((pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y));

      matches.push_back(
         Match(
            Point2d(matchRegions[i].ellipse.center.x, matchRegions[i].ellipse.center.y),
            Point2d(refRegions[TCmaxIdx].ellipse.center.x, refRegions[TCmaxIdx].ellipse.center.y),
            TCmaxIdx, i,
            cmf, distance, TCmaxCount));
   }
}

double getMatchScore(const RotatedRect& refEllipse, const RotatedRect& matchEllipse, Mat& T, const ProgramSettings& settings)
{
//   typedef bg::model::d2::point_xy<double> pt_xy;
   typedef bg::model::polygon<bg::model::d2::point_xy<double>> polygon;

   double ellipseUnion, ellipseIntersect;
   double x, y;

   // transform match to reference
   double M11 = T.at<double>(0,0);
   double M12 = T.at<double>(0,1);
   double M13 = T.at<double>(0,2);
   double M21 = T.at<double>(1,0);
   double M22 = T.at<double>(1,1);
   double M23 = T.at<double>(1,2);
   double M31 = T.at<double>(2,0);
   double M32 = T.at<double>(2,1);
   double M33 = T.at<double>(2,2);
   
   // geometries for intersection
   polygon refPoly, matchPoly;
   deque<polygon> unionPoly;
   deque<polygon> intersectPoly;
   
   vector<Point2d> matchEllipsePts(64U);
   vector<Point2d> refEllipsePts(64U);

   // compute points
   getEllipsePoints(matchEllipsePts, matchEllipse, 64U);
   getEllipsePoints(refEllipsePts, refEllipse, 64U);

   // transform match to reference image
   for ( Point2d& pt : matchEllipsePts )
   {
      x = pt.x;
      y = pt.y;

      pt.x = (x*M11 + y*M12 + M13) / (x*M31 + y*M32 + M33);
      pt.y = (x*M21 + y*M22 + M23) / (x*M31 + y*M32 + M33);
   
      bg::append(matchPoly, bg::model::d2::point_xy<double>(pt.x,pt.y));
   }
   
   // append last point
   bg::append(matchPoly, bg::model::d2::point_xy<double>(matchEllipsePts[0].x, matchEllipsePts[0].y));

   // create match polygon
   for ( Point2d& pt : refEllipsePts )
      bg::append(refPoly, bg::model::d2::point_xy<double>(pt.x, pt.y));

   // append last point
   bg::append(refPoly, bg::model::d2::point_xy<double>(refEllipsePts[0].x, refEllipsePts[0].y));

   // find intersection and union
   bg::union_(refPoly, matchPoly, unionPoly);
   bg::intersection(refPoly, matchPoly, intersectPoly);

   ellipseUnion = 0;
   ellipseIntersect = 0;

   for ( polygon& p : unionPoly )
      ellipseUnion += bg::area(p);
   for ( polygon& p : intersectPoly )
      ellipseIntersect += bg::area(p);
   
   return ellipseIntersect / ellipseUnion;
}

void calcResults(Mat& output, Results& results, const ProgramSettings& settings,
   const vector<Match>& matches, const vector<Region>& refRegions,
   const vector<Region>& matchRegions, const Mat& refImage, const Mat& matchImage)
{
   // initialize output results
   output.create(max(refImage.size().height, matchImage.size().height),
                     refImage.size().width+matchImage.size().width,
                     CV_8UC3);

   // reference image
   refImage.copyTo(output(Rect(0,0,refImage.size().width, refImage.size().height)));
   matchImage.copyTo(output(Rect(refImage.size().width,0,matchImage.size().width, matchImage.size().height)));

   vector<bool> acceptMatch(matches.size());
   vector<bool> goodMatch(matches.size());
   
   int idx = 0;
   // filter bad matches
   for ( const Match& m : matches )
   {
      acceptMatch[idx] = (
           m.matchedLines > settings.descriptor.N * settings.descriptor.minMatches && 
           (( m.cmf <= settings.descriptor.maxCmf && m.cmf >= settings.descriptor.minCmf ) ||
              m.dist < settings.descriptor.maxDist ));
      idx++;
   }

   // Test homography transforms
   Mat T(3,3,CV_64FC1);
   ifstream fin(settings.homographyFile);
   for ( int row = 0; row < 3; ++row )
      for ( int col = 0; col < 3; ++col )
         fin >> T.at<double>(row,col);
   fin.close();

   T = T.inv();
   
   // transform points
   int goodMatches = 0;
   int totalMatches = 0;
   for ( size_t i = 0; i < matches.size(); ++i )
   {
      if ( !acceptMatch[i] )
         continue;

      RotatedRect matchEllipse = matchRegions[matches[i].matchIndex].ellipse;
      RotatedRect refEllipse = refRegions[matches[i].refIndex].ellipse;

      // matchEllipse.size.width /= settings.descriptor.ellipseSize;
      // matchEllipse.size.height /= settings.descriptor.ellipseSize;
      // refEllipse.size.width /= settings.descriptor.ellipseSize;
      // refEllipse.size.height /= settings.descriptor.ellipseSize;

      // compute intersection over union
      double matchScore = getMatchScore(refEllipse, matchEllipse, T, settings);
      
      if ( matchScore > 0.5 )
      {
         goodMatch[i] = true;
         goodMatches++;
      } else {
         goodMatch[i] = false;
      }

      totalMatches++;
   }

   if ( settings.saveImage || settings.showImage ) {
      // draw bad matches
      for ( int i = 0; i < goodMatch.size(); ++i ) {
         if ( acceptMatch[i] && !goodMatch[i] ) {
            double x = matches[i].points[0].x;
            double y = matches[i].points[0].y;

            line(output,
                 Point(x+refImage.size().width,y),
                 Point(matches[i].points[1].x,
                       matches[i].points[1].y),
                 Scalar(0,0,255),
                 2);
         }
      }

      // draw good matches
      for ( int i = 0; i < goodMatch.size(); ++i ) {
         if ( acceptMatch[i] && goodMatch[i] ) {
            bool skip = false;
            
            double x = matches[i].points[0].x;
            double y = matches[i].points[0].y;

            // TODO draw more random colors
            line(output,
                 Point(x+refImage.size().width,y),
                 Point(matches[i].points[1].x,
                       matches[i].points[1].y),
                 Scalar(0,255,0),
                 2);
         }
      }
   }

   results.correctMatches = goodMatches;
   results.incorrectMatches = totalMatches - goodMatches;
   results.accuracy = 100.0 * goodMatches / totalMatches;

   //cout << myCmf << "    " << results.correctMatches << ":" << results.incorrectMatches << ":" << results.accuracy << endl;
}

int main(int argc, char *argv[])
{
   ProgramSettings settings;

   // load settings
   if ( !parseSettings(argc, argv, settings) )
      return -1;

   srand(time(0));

   // local variables 
   Mat refImage = imread(settings.refImage);
   Mat matchImage = imread(settings.matchImage);
   Mat output;
   vector<Region> refRegions, matchRegions;
   vector<Match> matches;
   Results results;
   clock_t t;

   // extract features from reference image and matching image
   t = clock();
   extractRegions(refRegions, refImage, settings, true);
   results.refDetectTime = clock() - t;
   
   t = clock();
   extractRegions(matchRegions, matchImage, settings, false);
   results.matchDetectTime = clock() - t;

   // find matching regions
   t = clock();
   findMatches(matches, refRegions, matchRegions, settings);
   results.matchingTime = clock() - t;

   calcResults(output, results, settings, matches, refRegions, matchRegions, refImage, matchImage);

   // Output results
   cout << "       Accuracy: " << results.accuracy << "%" << endl;
   cout << "  Total Correct: " << results.correctMatches << endl;
   cout << "Total Incorrect: " << results.incorrectMatches << endl;

   cout << "Reference Detection time: " << setprecision(4) << fixed << (double)results.refDetectTime / CLOCKS_PER_SEC << "s" << endl;
   cout << "    Match Detection time: " << setprecision(4) << fixed << (double)results.matchDetectTime / CLOCKS_PER_SEC << "s" << endl;
   cout << "           Matching time: " << setprecision(4) << fixed << (double)results.matchingTime / CLOCKS_PER_SEC << "s" << endl;

   if ( settings.saveImage || settings.saveConfig ) {
      string outputName = buildOutputString(settings.outputLocation, results);

      if ( settings.saveImage ) {
         cout << "Writing output image to " << outputName << ".jpg" << endl;
         imwrite(outputName + ".jpg", output);
      }

      if ( settings.saveConfig )
         cout << "Writing config file to " << outputName << ".cfg" << endl;
         writeConfig(outputName + ".cfg", settings); 
   }

   if ( settings.showImage ) {
      imshow("Output", output);
      waitKey(0);
   }

   return 0;
}


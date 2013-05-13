// TODO List:
//    Compute BitWise descriptor
//    Convert FLANN to do hamming distance
//    Compute P.R. Curves
// Wish List:
//    Try other feature points (SIFT)
//    

#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include <iostream>
#include <tuple>
#include <limits>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <deque>
#include <random>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "settings.h"
#include "results.h"

using namespace cv;
using namespace std;

namespace bg = boost::geometry;

//#define CUBIC_INTERPOLATION

// Used for debugging
//#define DRAW_CIRCLE
//#define DRAW_RES

typedef unsigned char BinaryMatchType;
const int CV_BTYPE = CV_8UC1;

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
   
// coefficients for transforming unit circle to ellipse
struct Coefs{
   double a, b, t;
}; 

struct Region {
   RotatedRect ellipse;
   vector<Line> lines;
   vector<Mat> descriptors;
   vector<double> mean;
   vector<double> err;

   Coefs coefs;

   long int baseIdx;

   double meanErr;
};

// TODO Add prototypes

void getEllipsePoints(vector<Point2d>& vertices, const RotatedRect& box, size_t points=1000U);
void drawEllipse(Mat& image, const RotatedRect& box, const Scalar& color = Scalar(0,0,255));
void drawEllipse(Mat& image, const vector<Point2d>& vertices, const Scalar& color = Scalar(0,0,255));
void calcResults(Mat& output, Results& results, const ProgramSettings& settings,
   const vector<Match>& matches, const vector<Region>& refRegions,
   const vector<Region>& matchRegions, const Mat& refImage, const Mat& matchImage);

#if 0
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
#endif

void computeEllipseCoef(const RotatedRect& box, Coefs& coefs)
{
   // from point within unit circle r1, t1
   // t2 = t1+t;
   // r2 = r1*a*b / sqrt(b*b*cos(t1-M_PI/2)*cos(t1-M_PI/2) + a*a*sin(t1-M_PI/2)*sin(t1-M_PI/2))

   RotatedRect r = box;
   if ( r.size.height < r.size.width )
   {
      r.size.height = box.size.width;
      r.size.width = box.size.height;

      r.angle = (box.angle + 90.0);
   }
   
   coefs.t = r.angle * M_PI / 180.0f;
   coefs.a = r.size.height / 2.0;
   coefs.b = r.size.width / 2.0;
}

// transform a radius and angle from the unit circle to the ellipse described by a, b, and t.
// get a, b, and t from computeEllipseCoef function (assumes ellipse centered at 0,0)
void ellipseTransform(double r1, double t1, const Coefs& coefs, double& r2, double& t2)
{
   double bCos = coefs.b * cos(t1-M_PI/2);
   double aSin = coefs.a * sin(t1-M_PI/2);

   t2 = t1 + coefs.t;
   r2 = r1 * coefs.a * coefs.b / sqrt(bCos * bCos + aSin * aSin); 
}

// Used to approximate intersection of two ellipses
void getEllipsePoints(vector<Point2d>& vertices, const RotatedRect& box, size_t points)
{
   Coefs coefs;
   computeEllipseCoef(box, coefs);

   double step = 2 * M_PI / points;
   double theta = 2 * M_PI;

   double radius, ang;

   // clockwise to make area work with boost
   vertices.resize(points);
   for ( size_t i = 0; i < points; ++i )
   {
      ellipseTransform(1.0, theta, coefs, radius, ang);
      vertices[i] = Point2f(
         box.center.x + radius * cos(ang),
         box.center.y + radius * sin(ang));
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

#ifdef DRAW_CIRCLE
   //ellipse(image, box, Scalar(255,0,0));

   Point2f verts[4];
   box.points(verts);
   for ( int i = 0; i < 4; ++i )
      line(image, verts[i], verts[(i+1)%4], Scalar(255,0,0));
#endif
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

double polarDist(double r1, double r2, double t1, double t2)
{
   return sqrt( r1*r1 + r2*r2 - 2*r1*r2*cos(t1-t2) );
}

bool minDistCheck( double r1, double r2, double t1, double t2, const ProgramSettings& settings )
{
   if ( settings.descriptor.minDist <= 0 )
      return r1 != r2 || t1 != t2;
   return polarDist(r1, r2, t1, t2) > settings.descriptor.minDist;
}

double randomUniform(double min, double max)
{
   static default_random_engine gen(time(0));
   uniform_real_distribution<double> uDist(min, max);
   return uDist(gen);
}

double randomNormal(double mean, double stdDev)
{
   static default_random_engine gen(time(0));
   normal_distribution<double> nDist(mean, stdDev);
   return nDist(gen);
}

// generate two random points in a given region (assumes the coefficients in the region are already computed)
void genRandomPoints(const Point2f& center, const Coefs& coefs, Line& linePts, const ProgramSettings& settings)
{
   double r1, r2, t1, t2;
   do {
      // generate random points in a unit circle (working in polar coords, convert later)
      if ( settings.descriptor.type == ProgramSettings::DescriptorSettings::EDGE_POINTS ) {
         // pick random points on edge of circle
         t1 = randomUniform(0.0, 2.0*M_PI);
         t2 = t1 - M_PI;
         //t2 = randomUniform(0.0, 2.0*M_PI);
         r1 = r2 = 1.0;

      } else if ( settings.descriptor.type == ProgramSettings::DescriptorSettings::RAND_POINTS_UNIFORM ) {
         // http://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
         // compute random point in unit circle (uniform distribution)
         t1 = randomUniform(0.0, 2.0*M_PI);
         t2 = randomUniform(0.0, 2.0*M_PI);
         
         r1 = randomUniform(0.0, 1.0) + randomUniform(0.0, 1.0);
         r2 = randomUniform(0.0, 1.0) + randomUniform(0.0, 1.0);

         if ( r1 > 1.0 )
            r1 = 2 - r1;
         if ( r2 > 1.0 )
            r2 = 2 - r2;
      } else if ( settings.descriptor.type == ProgramSettings::DescriptorSettings::RAND_POINTS_GAUSSIAN ) {
         // compute random points in unit circle (normal distribution)
         t1 = randomUniform(0.0, 2*M_PI);
         t2 = randomUniform(0.0, 2*M_PI);

         r1 = min(1.0, fabs(randomNormal(0.0, settings.descriptor.gaussStdDev)));
         r2 = min(1.0, fabs(randomNormal(0.0, settings.descriptor.gaussStdDev)));
      }
   } while ( !minDistCheck(r1, t1, r2, t2, settings) );
        
   // map to ellipse
   ellipseTransform(r1, t1, coefs, r1, t1);
   ellipseTransform(r2, t2, coefs, r2, t2);

   // convert to rectangular
   linePts.first = Point2d(r1*cos(t1) + center.x,
                           r1*sin(t1) + center.y);
   
   linePts.second = Point2d(r2*cos(t2) + center.x,
                            r2*sin(t2) + center.y);

   // TODO return a bool representing if points were found
}

void buildLineDescriptor(const Mat& image, Region& region, size_t regionIdx, const ProgramSettings& settings)
{
   double sum, diffs;
      
   Point2d& pt1 = region.lines[regionIdx].first;
   Point2d& pt2 = region.lines[regionIdx].second;

   // create descriptor
   region.descriptors[regionIdx].create(Size(settings.descriptor.l, 1), CV_FTYPE);

   // linear interpolation of samples
   if ( settings.descriptor.interpSamples ) {
      // create line iterator (4-connected line works better than 8)
      LineIterator lineIt(image, pt1, pt2, 4);

      int lineLen = lineIt.count;   // number of pixels in line

      // copy line to vector for iteration
      Mat lineMat(1, lineLen, CV_FTYPE);
      for ( int i = 0; i < lineIt.count; ++i, ++lineIt )
         lineMat.at<FType>(0,i) = (FType)(**lineIt);

      // set first value
      region.descriptors[regionIdx].at<FType>(0,0) = lineMat.at<FType>(0,0);
      
      sum = lineMat.at<FType>(0,0);
      diffs = 0;
      
      // linear interpolate performed between points
      for ( int i = 1; i < settings.descriptor.l; ++i )
      {
         double x = (double)i * (lineLen-1.0) / (settings.descriptor.l-1.0);
         if ( fmod(x,1.0) == 0.0 ) {
            region.descriptors[regionIdx].at<FType>(0,i) = lineMat.at<FType>(0,(int)x);
         } else {
            double x1Val = lineMat.at<FType>(0,(int)x);
            double x2Val = lineMat.at<FType>(0,((int)x)+1);
          
            x = fmod(x,1.0);   // floating point part of x

            region.descriptors[regionIdx].at<FType>(0,i) = x1Val + x * (x2Val - x1Val);
         }

         diffs += fabs(region.descriptors[regionIdx].at<FType>(0,i) - region.descriptors[regionIdx].at<FType>(0,i-1));
         sum += region.descriptors[regionIdx].at<FType>(0,i);
      }
   // nearest neighbor sample
   } else {

      Point2d sample = pt1;
      Point2d step((pt2.x - pt1.x) / (settings.descriptor.l - 1.0),
                   (pt2.y - pt1.y) / (settings.descriptor.l - 1.0));

      sum = 0.0;
      diffs = 0.0;

      // set first value
      region.descriptors[regionIdx].at<FType>(0,0) = image.at<FType>(pt1);

      for ( int i = 1; i < settings.descriptor.l; ++i )
      {
         // increment step
         sample.x += step.x;
         sample.y += step.y;

         // get next sample
         region.descriptors[regionIdx].at<FType>(i) = image.at<unsigned char>(Point2i(sample.x, sample.y));
         sum += region.descriptors[regionIdx].at<FType>(i);

         diffs += fabs(region.descriptors[regionIdx].at<FType>(i-1) - 
                       region.descriptors[regionIdx].at<FType>(i));
      }
   }
   region.err[regionIdx] = diffs / (FType)(settings.descriptor.l - 1);
   region.mean[regionIdx] = sum / (FType)settings.descriptor.l;

   region.descriptors[regionIdx] -= region.mean[regionIdx];

   // If binary descriptor compare to mean and make the descriptor binary
   if ( settings.descriptor.binary ) {
      Mat desc(Size(settings.descriptor.l, 1), CV_BTYPE);

      int len = min((BinaryMatchType)(sizeof(BinaryMatchType)*8), (BinaryMatchType)settings.descriptor.l);

      for ( int i = 0; i < settings.descriptor.l; ++i ) {
         if ( region.descriptors[regionIdx].at<FType>(i) > 0.0 )
            desc.at<BinaryMatchType>(i) = 1;
         else
            desc.at<BinaryMatchType>(i) = 0;
      }
      desc.copyTo(region.descriptors[regionIdx]);
   }
}

void warpImage(const Mat& input, const RotatedRect& rect, int ksize, Mat& output, const ProgramSettings& settings)
{
   Point2f src[4];
   rect.points(src);

   // assumes output image is square
   const int imageSize = output.size().width;
   const int border = ksize/2;

   Point2f dst[3] = {Point2f(border, imageSize - border),
                     Point2f(border, border),
                     Point2f(imageSize - border, border)};

   Mat transform = getAffineTransform(src, dst);

#ifdef CUBIC_INTERPOLATION
   warpAffine(input, output, transform, output.size(), INTER_CUBIC, BORDER_REFLECT);
#else
   warpAffine(input, output, transform, output.size(), INTER_LINEAR, BORDER_REFLECT);
#endif
}

void smoothImage(Mat& image, double stdDev, int ksize)
{
   if ( stdDev > 0.0 ) {
      int ksize = stdDev * 4;
      if ( ksize % 2 == 0 ) ksize++;
      GaussianBlur(image, image, Size(ksize, ksize), stdDev);
   }
}

void buildFeatures(Region& region, const Mat& image, size_t lineCount, const ProgramSettings& settings)
{
   RotatedRect warpedRect;
   Coefs warpedCoefs;
   Mat warpedImage(255,255,image.type());

   // compute coefficients for ellipse mapping from unit circle
   if ( settings.descriptor.smoothRegion ) {
      int ksize = settings.descriptor.smoothing * 4;
      if ( ksize % 2 == 0 ) ksize++;

      // compute affine transformed image
      warpImage(image, region.ellipse, ksize, warpedImage, settings);
      smoothImage(warpedImage, settings.descriptor.smoothing, ksize);

      // compute coefficients to warp to this circle
      warpedRect = RotatedRect(Point2f(warpedImage.size().width/2, warpedImage.size().height/2),
         Size(warpedImage.size().width - 2*(ksize-1), warpedImage.size().height - 2*(ksize-1)), 0);

      computeEllipseCoef(warpedRect, warpedCoefs);
   } else {
      computeEllipseCoef(region.ellipse, region.coefs);
   }

   // resize vectors
   region.descriptors.resize(lineCount);
   region.mean.resize(lineCount);
   region.err.resize(lineCount);
   region.lines.resize(lineCount);

   // compute descriptor for each line
   for ( size_t idx = 0; idx < lineCount; ++idx ) {
      if ( settings.descriptor.smoothRegion ) {
         genRandomPoints(warpedRect.center, warpedCoefs, region.lines[idx], settings);
         buildLineDescriptor(warpedImage, region, idx, settings);
         region.meanErr += region.err[idx];
      } else {
         // sample region without computing affine transform
         genRandomPoints(region.ellipse.center, region.coefs, region.lines[idx], settings);
         buildLineDescriptor(image, region, idx, settings);
         region.meanErr += region.err[idx];
      }
   }

   region.meanErr /= (double)lineCount;

#ifdef DRAW_CIRCLE
   Mat img;
   cvtColor(image, img, CV_GRAY2RGB);
   drawEllipse(img, region.ellipse, Scalar(0,255,0));
   if ( !settings.descriptor.smoothRegion )
      drawLines(img, region.lines, Scalar(0,0,255));
   imshow("Image", img);
   
   if ( settings.descriptor.smoothRegion ) {
      cvtColor(warpedImage, img, CV_GRAY2RGB);
      drawEllipse(img, warpedRect, Scalar(0,255,0));
      imshow("ImageWin", img);
      waitKey(0);
      drawLines(img, region.lines, Scalar(0,0,255));
      imshow("ImageWin", img);
   }
   waitKey(0);
#endif
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
      if ( referenceImage ) {
         buildFeatures(regions[i], grayImage, s.Nk, settings); 
      } else {
         buildFeatures(regions[i], grayImage, s.N, settings); 
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
   std::vector<DMatch> matchList;
   size_t Nk = settings.descriptor.Nk;
   size_t N = settings.descriptor.N;
   size_t l = settings.descriptor.l;
   
   Mat refDescriptors, matchDescriptors;
   
   if ( settings.descriptor.binary ) {
      // add this as a template type later with 32 or 16 or 8
      //typedef cvflann::Hamming<_t> Distance_H32;
      //flann::GenericIndex<Distance_H32> *matcher;
      
      // initialize matrices
      refDescriptors.create(refRegions.size()*Nk, settings.descriptor.l, CV_BTYPE);
      matchDescriptors.create(matchRegions.size()*N, settings.descriptor.l, CV_BTYPE);

      // copy descriptors over
      for ( size_t i = 0; i < refRegions.size(); ++i )
         for ( size_t j = 0; j < Nk; ++j )
            refRegions[i].descriptors[j].copyTo(
               refDescriptors.row(i*Nk+j));
      
      for ( size_t i = 0; i < matchRegions.size(); ++i )
         for ( size_t j = 0; j < N; ++j )
            matchRegions[i].descriptors[j].copyTo(
               matchDescriptors.row(i*N+j));
      
      // TODO checks is a parameters to SearchParams which could be set by a setting (default is 32)
//      cvflann::KDTreeIndexParams iParams(settings.descriptor.kdTrees);
      /*
      matcher = new flann::GenericIndex<Distance_H32>(refDescriptors, iParams);

      const int knn = 2;

      Mat indices(matchRegions.size()*N, knn, CV_32SC1);
      Mat dists(matchRegions.size()*N, knn, CV_32FC1);
      
      cvflann::SearchParams sParams;
      matcher->knnSearch(matchDescriptors, indices, dists, knn, sParams);
      */

//      FlannBasedMatcher matcher(new flann::LshIndexParams(10, 12, 2));
      BFMatcher matcher(NORM_HAMMING);
      matcher.match(matchDescriptors, refDescriptors, matchList);
   } else {
      // initialize matrices
      refDescriptors.create(refRegions.size()*Nk, l, CV_FTYPE);
      matchDescriptors.create(matchRegions.size()*N, l, CV_FTYPE);

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
      matcher.match(matchDescriptors, refDescriptors, matchList);
   }
   
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

      //cout << distance << ' ' << cmf << ' ' << TCmaxCount << endl;
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
   const vector<Region>& matchRegions, const Mat& refImage, const Mat& matchImage )
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
           (( m.cmf <= settings.descriptor.maxCmf && m.cmf >= settings.descriptor.minCmf )));/* ||
              m.dist < settings.descriptor.maxDist ));*/
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

#ifdef DRAW_RES
      drawEllipse(output, matchEllipse, Scalar(255,0,0));
      drawEllipse(output, refEllipse, Scalar(0,255,0));
      imshow("Res",output);
      waitKey(0);
#endif

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


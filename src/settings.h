#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>
#include "results.h"

struct Results;

const std::string DEFAULT_CONFIG = "bin/config.cfg";

struct ProgramSettings {
   struct MserSettings {
      MserSettings() :
         delta(1),
         useRelativeArea(false),
         minArea(200),
         maxArea(14400),
         maxVariation(1.0),
         minDiversity(0.0),
         maxRegions(1024)
      {}

      int delta;
      
      bool useRelativeArea;
      double minArea;
      double maxArea;
      double maxVariation;
      double minDiversity;
      
      int maxRegions;   // maximum number of detectable regions
   };
   
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

   
   struct DescriptorSettings {
      DescriptorSettings() :
         ellipseSize(2.0),
         ellipsePoints(512U),
         //l(16),      // 16 for graf
         //Nk(250),    // 250 for graf
         //N(100),     // 100 for graf
         l(32),
         Nk(350),
         N(300),
         kdTrees(4),
         minDist(0.01),
         minCmf(0.6),
         maxDist(5.0),
         minMatches(0.05),
         minMeanErr(0.0),
         k1(0.6),
         p1(0.7),
         p2(0.4),
         w1(1.0),
         w2(1.0),
         smoothRegion(false),
         smoothing(2.5)
      {}

      double ellipseSize;  // scale factor of ellipse to min bounding box
      size_t ellipsePoints;   // number of points to sample around perimeter of ellipse

      int l;   // number of points per line
      int Nk;  // number of lines per region in reference region
      int N;   // number of lines per region in test region

      int kdTrees;   // number of KDTrees used in FLANN

      double minDist; // minimum allowed length of line (with respect to sqrt(area) of region)

      double minCmf;  // minimum acceptable CMF score for matching
      double maxDist; // maximum distance two regions can be for the CMF score not to be considered
      double minMatches; // minimum ratio of matches that must occur for region to be accepted
      double minMeanErr; // minimum mean err for a region to be considered

      double k1;  // Dmax weighting (0,1]
      double p1;  // controls shape of confidence function
      double p2;  // controls shape of confidence function
      double w1;  // weight for distance
      double w2;  // weight for error

      bool smoothRegion;   // smooth the region? (smoothing doesn't matter if this is false)
      double smoothing; // strength of smoothing (relative to scale of region)
   };

   MserSettings mser;
   
   DescriptorSettings descriptor;

   std::string refImage;
   std::string matchImage;
   std::string homographyFile;
   bool showImage;
   bool saveImage;
   bool saveConfig;
   std::string outputLocation;
};

std::string buildOutputString(const std::string& format, const Results& results);
void writeConfig(const std::string& filename, ProgramSettings& settings);
bool parseSettings(int argc, char* argv[], ProgramSettings& settings);

#endif


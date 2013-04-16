#ifndef SETTINGS_H
#define SETTINGS_H

struct ProgramSettings {
   struct MserSettings {
      MserSettings() :
         delta(5),
         minArea(60),
         maxArea(14400),
         maxVariation(0.4),
         minDiversity(0.2)
      {}

      int delta;
      double minArea;
      double maxArea;
      double maxVariation;
      double minDiversity;
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
         l(10),
         Nk(450),
         N(450),
         minDist(0.25),
         k1(0.6),
         p1(1.0),
         p2(1.0),
         w1(1.0),
         w2(1.0),
         smoothing(2.5)
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

#endif


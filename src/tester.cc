#include "matcher.h"
#include <cmath>

//#define MAX_CMF
#define MIN_MATCHES

#define COUT_ENABLE

int computeCorrespondencies(const vector<Region>& matchRegions, const vector<Region>& refRegions, string homography)
{
   Mat T(3,3,CV_64FC1);
   ifstream fin(homography);
   for ( int row = 0; row < 3; ++row )
      for ( int col = 0; col < 3; ++col )
         fin >> T.at<double>(row,col);
   fin.close();

   T = T.inv();
   
   // takes some time only do this once
   int corr = 0;
   for (size_t i = 0; i < refRegions.size(); ++i )
      for (size_t j = 0; j < matchRegions.size(); ++j )
         if ( getMatchScore(refRegions[i].ellipse, matchRegions[j].ellipse, T, ProgramSettings()) > 0.5 ) {
            corr++;
            break;
         }

   return corr;
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

   int corr = computeCorrespondencies(matchRegions, refRegions, settings.homographyFile);
   
#ifdef COUT_ENABLE
   cout << "Correspondencies: " << corr << endl; 
#endif

   matches.clear();

   // find matching regions
   t = clock();
   findMatches(matches, refRegions, matchRegions, settings);
   results.matchingTime = clock() - t;

#ifdef COUT_ENABLE
   cout << "Reference Detection time: " << setprecision(4) << fixed << (double)results.refDetectTime / CLOCKS_PER_SEC << "s" << endl;
   cout << "    Match Detection time: " << setprecision(4) << fixed << (double)results.matchDetectTime / CLOCKS_PER_SEC << "s" << endl;
   cout << "           Matching time: " << setprecision(4) << fixed << (double)results.matchingTime / CLOCKS_PER_SEC << "s" << endl;
#endif

   vector<double> precision;
   vector<double> recall;

   double oldCmf = settings.descriptor.maxCmf;
   double oldMinMatches = settings.descriptor.minMatches;
#ifdef MAX_CMF
   settings.descriptor.minMatches = 0.0;
   for ( double maxCmf = oldCmf * 20; maxCmf >= oldCmf; maxCmf -= oldCmf )
   {
      settings.descriptor.maxCmf = maxCmf;
      
      calcResults(output, results, settings, matches, refRegions, matchRegions, refImage, matchImage);

      if ( results.correctMatches + results.incorrectMatches == 0 )
         precision.push_back(0.0);
      else
         precision.push_back((double)results.incorrectMatches / (results.correctMatches + results.incorrectMatches));
      
      if ( corr == 0 )
         recall.push_back(0.0);  // shouldn't happen
      else
         recall.push_back(results.correctMatches / (double)corr);

#ifdef COUT_ENABLE
      // Output results
      cout << "     minMatches: " << settings.descriptor.minMatches << endl;
      cout << "         maxCMF: " << settings.descriptor.maxCmf << endl;
      cout << "       Accuracy: " << results.accuracy << "%" << endl;
      cout << "  Total Correct: " << results.correctMatches << endl;
      cout << "Total Incorrect: " << results.incorrectMatches << endl;
      cout << "      Precision: " << precision.back() << endl;
      cout << "         Recall: " << recall.back() << endl;
      cout << endl;
#endif      
      if ( settings.showImage ) {
         imshow("Output", output);
         waitKey(0);
      }
   }
#endif

#ifdef MIN_MATCHES
   //settings.descriptor.maxCmf = oldCmf;
   for ( double minMatches = 0.0; minMatches < 0.9; minMatches+=0.03333 ) 
   {
      settings.descriptor.minMatches = minMatches;

      calcResults(output, results, settings, matches, refRegions, matchRegions, refImage, matchImage);

      if ( results.correctMatches + results.incorrectMatches == 0 )
         precision.push_back(0.0);
      else
         precision.push_back((double)results.incorrectMatches / (results.correctMatches + results.incorrectMatches));
      
      if ( corr == 0 )
         recall.push_back(0.0);  // shouldn't happen
      else
         recall.push_back(results.correctMatches / (double)corr);
#ifdef COUT_ENABLE
      // Output results
      cout << "     minMatches: " << settings.descriptor.minMatches << endl;
      cout << "         maxCMF: " << settings.descriptor.maxCmf << endl;
      cout << "       Accuracy: " << results.accuracy << "%" << endl;
      cout << "  Total Correct: " << results.correctMatches << endl;
      cout << "Total Incorrect: " << results.incorrectMatches << endl;
      cout << "      Precision: " << precision.back() << endl;
      cout << "         Recall: " << recall.back() << endl;
      cout << endl;
#endif 
      if ( settings.showImage ) {
         imshow("Output", output);
         waitKey(0);
      }
   }
#endif

   cout << "Writing precision recal information to PR.csv" << endl;
   ofstream fout("PR.csv");
   for ( size_t i; i < precision.size(); ++i ) {
      fout << precision[i];
      fout << ',';
      fout << recall[i];
      fout << endl;
   }
   fout.close();

   return 0;
}


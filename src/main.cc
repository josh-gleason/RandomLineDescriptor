#include "matcher.h"

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

      if ( settings.saveConfig ) {
         cout << "Writing config file to " << outputName << ".cfg" << endl;
         writeConfig(outputName + ".cfg", settings); 
      }
   }

   if ( settings.showImage ) {
      imshow("Output", output);
      waitKey(0);
   }

   return 0;
}


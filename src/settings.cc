#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include "settings.h"

namespace po = boost::program_options;

using namespace std;

string buildOutputString(const string& format, const Results& results)
{
   istringstream sin(format);
   ostringstream sout;

   int n = -1;

   time_t rawtime;
   tm* timeinfo;
   char buffer[25];
   
   time(&rawtime);
   timeinfo = localtime(&rawtime);

   strftime(buffer,25,"%m %d %Y %H %M %S", timeinfo);
   
   string whole = buffer;
   string month, day, year, hour, minute, second;

   // get asctime
   istringstream sinTime(whole);
   sinTime >> month >> day >> year >> hour >> minute >> second;

   int count = format.size();
   char val;
   while ( count > 0 ) {
      sin >> val;
      count--;
      if ( val == '%' && count > 0 ) {
         sin >> val;
         count--;
         if ( val == 'a' ) {
            sout << setprecision(2) << fixed << results.accuracy;
         } else if ( val == 'M' ) {
            sout << month;
         } else if ( val == 'd' ) {
            sout << day;
         } else if ( val == 'y' ) {
            sout << year;
         } else if ( val == 'h' ) {
            sout << hour;
         } else if ( val == 'm' ) {
            sout << minute;
         } else if ( val == 's' ) {
            sout << second;
         } else if ( val == 'c' ) {
            sout << results.correctMatches;
         } else if ( val == 'i' ) {
            sout << results.incorrectMatches;
         } else if ( val == '%' ) {
            sout << '%';
         } else {
            // not found write to file name
            sout << '%' << val;
         }
      } else {
         sout << val;
      }
   }
   
   return sout.str();
}

void writeConfig(const string& filename, ProgramSettings& settings)
{
   ofstream fout(filename.c_str());
   
   ProgramSettings& s = settings;
   ProgramSettings::DescriptorSettings& d = settings.descriptor;
   ProgramSettings::MserSettings& m = settings.mser;

   fout << "# Automatically generated config file" << endl;
   fout << "REF_IMG = " << s.refImage << endl;
   fout << "MAT_IMG = " << s.matchImage << endl;
   fout << "HOMOG = " << s.homographyFile << endl;
   fout << "SHOW_IMG = " << s.showImage << endl;
   fout << "SAVE_IMG = " << s.saveImage << endl;
   fout << "SAVE_CONFIG = " << s.saveConfig << endl;
   fout << "OUT_LOC = " << s.outputLocation << endl;
   fout << "MSER_DELTA = " << m.delta << endl;
   fout << "MSER_USE_RELATIVE_AREA = " << m.useRelativeArea << endl;
   fout << "MSER_MIN_AREA = " << m.minArea << endl;
   fout << "MSER_MAX_AREA = " << m.maxArea << endl;
   fout << "MSER_MAX_VARIATION = " << m.maxVariation << endl;
   fout << "MSER_MIN_DIVERSITY = " << m.minDiversity << endl;
   fout << "MSER_MAX_REGIONS = " << m.maxRegions << endl;
   fout << "DES_ELLIPSE_SIZE = " << d.ellipseSize << endl;
   fout << "DES_ELLIPSE_POINTS = " << d.ellipsePoints << endl;
   fout << "DES_L = " << d.l << endl;
   fout << "DES_NK = " << d.Nk << endl;
   fout << "DES_N = " << d.N << endl;
   fout << "DES_MIN_DIST = " << d.minDist << endl;
   fout << "DES_SMOOTH = " << d.smoothRegion << endl;
   fout << "DES_SMOOTH_SIGMA = " << d.smoothing << endl;
   fout << "DES_MIN_MEAN_ERR = " << d.minMeanErr << endl;
   fout << "MAT_MIN_MATCHES = " << d.minMatches << endl;
   fout << "MAT_MIN_CMF = " << d.minCmf << endl;
   fout << "MAT_MAX_CMF = " << d.maxCmf << endl;
   fout << "MAT_MAX_DIST = " << d.maxDist << endl;
   fout << "MAT_K1 = " << d.k1 << endl;
   fout << "MAT_P1 = " << d.p1 << endl;
   fout << "MAT_W1 = " << d.w1 << endl;
   fout << "MAT_P2 = " << d.p2 << endl;
   fout << "MAT_W2 = " << d.w2 << endl;
   fout << "DES_KDTREES = " << d.kdTrees << endl;

   fout.close();
}

bool parseSettings(int argc, char* argv[], ProgramSettings& settings)
{
   string configFile;
      
   po::options_description generic("Allowed options");
   generic.add_options()
      ("help", "produce help message")
      ("CONFIG_FILES,c", po::value<string>(&configFile)->default_value(DEFAULT_CONFIG),
         "Configuration file.")
   ;

   ProgramSettings& s = settings;
   ProgramSettings::DescriptorSettings& d = settings.descriptor;
   ProgramSettings::MserSettings& m = settings.mser;

   po::options_description config("Allowed options");
   config.add_options()
      ("REF_IMG", po::value<string>(&s.refImage)->required(),
         "Reference image file path.")
      ("MAT_IMG", po::value<string>(&s.matchImage)->required(),
         "Matching image file path.")
      ("HOMOG", po::value<string>(&s.homographyFile)->required(),
         "File containing homography transform which maps matching image to reference image.")
      ("SHOW_IMG", po::value<bool>(&s.showImage)->required(),
         "Show the resulting matches after results are computed.")
      ("SAVE_IMG", po::value<bool>(&s.saveImage)->required(),
         "Save the resulting image after results are computed.")
      ("SAVE_CONFIG", po::value<bool>(&s.saveConfig)->required(),
         "Save the resulting config file after the results are computed.")
      ("OUT_LOC", po::value<string>(&s.outputLocation)->required(),
         "Output image or config file name, use the following flags in the file name...\n"
         "\t%a Result accuracy\n"
         "\t%M %d %y Month, day, year\n"
         "\t%h %m %s Hour, minute, second.\n")
      ("MSER_DELTA", po::value<int>(&m.delta)->required(),
         "")
      ("MSER_USE_RELATIVE_AREA", po::value<bool>(&m.useRelativeArea)->required(),
         "")
      ("MSER_MIN_AREA", po::value<double>(&m.minArea)->required(),
         "")
      ("MSER_MAX_AREA", po::value<double>(&m.maxArea)->required(),
         "")
      ("MSER_MAX_VARIATION", po::value<double>(&m.maxVariation)->required(),
         "")
      ("MSER_MIN_DIVERSITY", po::value<double>(&m.minDiversity)->required(),
         "")
      ("MSER_MAX_REGIONS", po::value<int>(&m.maxRegions)->required(),
         "Max number of regions per image.")
      ("DES_ELLIPSE_SIZE", po::value<double>(&d.ellipseSize)->required(),
         "")
      ("DES_ELLIPSE_POINTS", po::value<size_t>(&d.ellipsePoints)->required(),
         "")
      ("DES_L", po::value<int>(&d.l)->required(),
         "")
      ("DES_NK", po::value<int>(&d.Nk)->required(),
         "")
      ("DES_N", po::value<int>(&d.N)->required(),
         "")
      ("DES_MIN_DIST", po::value<double>(&d.minDist)->required(),
         "")
      ("DES_SMOOTH", po::value<bool>(&d.smoothRegion)->required(),
         "")
      ("DES_SMOOTH_SIGMA", po::value<double>(&d.smoothing)->required(),
         "")
      ("DES_MIN_MEAN_ERR", po::value<double>(&d.minMeanErr)->required(),
         "")
      ("MAT_MIN_MATCHES", po::value<double>(&d.minMatches)->required(),
         "")
      ("MAT_MIN_CMF", po::value<double>(&d.minCmf)->required(),
         "")
      ("MAT_MAX_CMF", po::value<double>(&d.maxCmf)->required(),
         "")
      ("MAT_MAX_DIST", po::value<double>(&d.maxDist)->required(),
         "")
      ("MAT_K1", po::value<double>(&d.k1)->required(),
         "")
      ("MAT_P1", po::value<double>(&d.p1)->required(),
         "")
      ("MAT_W1", po::value<double>(&d.w1)->required(),
         "")
      ("MAT_P2", po::value<double>(&d.p2)->required(),
         "")
      ("MAT_W2", po::value<double>(&d.w2)->required(),
         "")
      ("DES_KDTREES", po::value<int>(&d.kdTrees)->required(),
         "")
   ;

   // parse for help file and config file
   try {
      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, generic), vm);

      // check for help first
      if ( vm.count("help") )
      {
         cout << generic << endl
              << config << endl;
         return false;
      }
      
      po::notify(vm);
   } catch (std::exception& e) {
      cout << "Parsing Error: " << e.what() << endl;
      return false;
   }

   try {
      ifstream fin(configFile.c_str());
      if ( !fin.good() )
      {
         cout << "Config file " << configFile << " not found!" << endl;
         return false;
      }

      config.add(generic);

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, config), vm);
      po::store(po::parse_config_file(fin, config), vm);
      po::notify(vm);

      fin.close();
   } catch (std::exception& e) {
      cout << "Parsing Error: " << e.what() << endl;
      return false;
   }

   return true;
}

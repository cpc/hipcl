#include <iostream>
#include <vector>

#include <cxxopts.hpp>

#include "hipcl_config.h"

int main(int argc, char const *argv[]) {

  try {

    cxxopts::Options options("hipcl_config", "configuration helper for HipCL");

    std::string group("hipcl_options");
    options.add_options(group)("p,path", "print HIP_PATH (use env var if set, "
                                         "else determine from hipconfig path)")(
        "C,cpp_config", "print C++ compiler options")(
        "c,compiler", "print compiler")("P,platform", "print platform")(
        "v,version", "print hip version")("f,full", "print full config")(
        "h,help", "print help message with all options");

    //    print "  --full, -f         : print full config\n";
    //    print "  --check            : check configuration\n";
    //    print "  --newline, -n      : print newline\n";

    char **argv_copy = (char **)argv;
    auto result = options.parse(argc, argv_copy);

    // TODO these should take into account also CUDA/HIP/HCC but currently don't
    const char *hip_path = CMAKE_INSTALL_PREFIX;
    const char *hip_cpp_options =
        "-D__HIP_PLATFORM_HIPCL__=  " HIPCL_CXX_OPTIONS;
    const char *hip_platform = HIP_PLATFORM;
    const char *hip_version = HIPCL_VERSION_FULL;

    if (result.count("p") > 0) {
      std::cout << hip_path;
      return 0;
    }
    if (result.count("C") > 0) {
      std::cout << hip_cpp_options;
      return 0;
    }

    if (result.count("c") > 0) {
      std::cout << hip_platform;
      return 0;
    }

    if (result.count("P") > 0) {
      std::cout << hip_platform;
      return 0;
    }

    if (result.count("v") > 0) {
      std::cout << hip_version;
      return 0;
    }

    std::vector<std::string> groups;
    groups.push_back("hipcl_options");
    std::cout << options.help(groups);

    return (result.count("h") > 0) ? 0 : 1;
  } catch (cxxopts::OptionSpecException &e) {
    std::cerr << "Error in specified options: " << e.what() << "\n";
    return 1;
  }

  catch (cxxopts::OptionParseException &e) {
    std::cerr << "Error while parsing options: " << e.what() << "\n";
    return 1;
  }
}


#include "glog/logging.h"
#include <iostream>
#include "gflags/gflags.h"
#include <sstream>
#include <vector>
#include <string>

using std::stringstream;
using std::string;
using std::vector;

DEFINE_int32(print, 1, "the print times.");
DEFINE_string(name, "tseting", "argvsdfsdf");


string Vector2String(const vector<string>& vocab, vector<int> in) {
    int i;
    stringstream ss;
    for (auto it = in.begin(); it != in.end(); it++) {
        ss << vocab[*it];
    }
    return ss.str();
}


int main(int argc, char** argv) {
    vector<string> vocab;
    vocab.push_back("你");
    vocab.push_back("好");
    vocab.push_back("世");
    vocab.push_back("界");

    vector<int> ind = {3,2,1,0,1,2,3};

    string out = Vector2String(vocab, ind);

    std::cout << "out: " << out << std::endl;

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FLAGS_log_dir = "./test_log";

    google::SetLogFilenameExtension(".log");
    google::InitGoogleLogging("AlgLog");

    LOG(INFO) << "hello from main";

    google::FlushLogFiles(google::GLOG_INFO);

    std::cout << "OK" << std::endl;

    std::cout << FLAGS_name << std::endl;

    for (int i=0; i<FLAGS_print; i++)
        std::cout << FLAGS_print << std::endl;


    gflags::ShutDownCommandLineFlags();

    

}
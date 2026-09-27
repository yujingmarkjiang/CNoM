#include "cnom/gate.hpp"
#include "cnom/router.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Source {
    std::string id;
    std::string camera;
    std::ifstream scores;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: cnom-edge CONFIG.scores.tsv CLOUD_URL\n"
                     "score file columns: camera_id source_id t_s score_0..score_7\n";
        return 2;
    }
    cnom::TemporalPolicy policy;
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }
    std::string line;
    std::getline(in, line);
    cnom::TemporalGate* gate = nullptr;
    std::string cur_cam, cur_src;
    std::vector<cnom::CandidateEvent> events;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cam, src;
        double t;
        ss >> cam >> src >> t;
        float q[8];
        for (int i = 0; i < 8; ++i) ss >> q[i];
        if (!gate || cam != cur_cam || src != cur_src) {
            if (gate) {
                auto tail = gate->flush(t);
                events.insert(events.end(), tail.begin(), tail.end());
                delete gate;
            }
            cur_cam = cam;
            cur_src = src;
            gate = new cnom::TemporalGate(cam, src, policy);
        }
        const float s = cnom::max_aggregate(q, 8);
        auto out = gate->update(t, s);
        events.insert(events.end(), out.begin(), out.end());
    }
    if (gate) {
        auto tail = gate->flush();
        events.insert(events.end(), tail.begin(), tail.end());
        delete gate;
    }
    std::printf("{\"cloud\":\"%s\",\"candidates\":%zu}\n", argv[2], events.size());
    for (const auto& e : events) {
        std::printf("%s %s %.3f %.3f %.3f %.4f\n", e.id.c_str(), e.camera_id.c_str(),
                    e.start_s, e.end_s, e.trigger_s, e.max_score);
    }
    return 0;
}

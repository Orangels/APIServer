//
// Created by xinxueshi on 2020/5/6.
//

#ifndef DPH_STRUCTS_H
#define DPH_STRUCTS_H

#include <vector>

using namespace std;

typedef struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
} Box;

typedef struct Angle {
    float Y;
    float P;
    float R;
} Angle;

typedef struct InstenceFace {
    long time;
    vector<float> face_fea;
    int frequency;
} InstenceFace;

typedef struct Tracks {
    int  track_id;
    int start_frame;
    int  end_frame;
    bool is_edge;
    vector<std::pair<int,int>> tracks_last;
    bool is_live;
} Tracks;

class Person_ID {
    public:
    Person_ID(){};
    ~Person_ID(){};
    vector<int>  person_id;
    vector<long long> timestamp;
    vector<std::string> paths;

};


#endif //DPH_STRUCTS_H

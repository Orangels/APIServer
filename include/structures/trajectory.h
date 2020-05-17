//
// Created by xxs on 2020/5/7.
//

#ifndef DHP_TRAJECT_H
#define DHP_TRAJECT_H
#include <iostream>
#include <vector>
#include "structures/structs.h"
using namespace std;

typedef struct Track_list {
    int track_id;
    int start_frame;
    int end_frame;
    bool is_live;
//    vector<int> traject;
    vector<pair<int,int>> traject;
} Track_list;


class Trajectory {

public:
    Trajectory(int dis);
    ~Trajectory();

    void update_trajectory(vector<int> ids,vector<Box> boxes_xxyy,int frame_count);
    void print_self();
    void merge_trajectory();

    //from python
    void update(vector<int> ids,vector<Box> boxes_xxyy,int frame_count);
    vector<vector<int>> groups;
    void merge_group();
    vector<int> merge_ids;
    void judge(int id,Box box);
    void judge_out(int id,Box box);
    int borders = 1920;
    int borders1 = 960;
    int borders2 = 2880;
    int distance =100;
    bool all_out(vector<int> ids,vector<Box> boxes_xxyy);
    int get_merged_id(int id);
    void remove_duplicates();

    vector<int> draw_id;
    vector<vector<int>> draw_list;
    vector<int> get_draw_list(int id);


    vector<Track_list> all_track_list;

    bool is_exit_id(int id);
    int get_order(int id);
    bool in_merge_arear(int y);

    vector<int> track_ids;
    vector<int> live_track_ids;

    vector<vector<pair<int,int>>> trajects;

    int merging_id;

    int Y = 1920;  //single first

    vector<int> merged_id;
    vector<int> merged_live_ids;
    vector<int> merged_start_frame;
    vector<int> merged_end_frame;
    vector<vector<pair<int,int>>> merged_traject;


};
#endif //DHP_TRACK_H
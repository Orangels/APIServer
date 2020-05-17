//
// Created by xinxuehsi  on 2020/5/1.
//
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "utils/match_id.h"
#include "box_tracking.h"
#include "utils/track.h"
#include "structures/structs.h"
#include "structures/trajectory.h"


Match_ID::Match_ID(int head_track_mistimes, int w, int h,int dis) {
    head_tracker = new Track(head_track_mistimes, w, h);
    trajectory = new Trajectory(dis);
}

Match_ID::~Match_ID() = default;

int  Match_ID::find_pid(int id){
    vector <int>::iterator iElement = find(match_tid.begin(),match_tid.end(),id);
    if( iElement != match_tid.end() )
	 {
	    int nPosition = distance(match_tid.begin(),iElement);
        return match_pid[nPosition];
	 }
	 else
	    return -1;
}
void  Match_ID::MergeResult(cv::Mat img, cv::Mat &result_img,int num, vector<int> ret1,vector<int>ret2, vector<int> ret3,int without_id,Person_ID * Person_id)
{
//    std::cout<<"merge"<<std::endl;
    vector<Box> boxes_xxyy;
    vector<int> ids;
    if(without_id){

        int mode = 1;
        Box box;
        for (int i=0; i< ret1.size(); i+=6)
        {
            if (ret1[i] ==mode)    //head
            {
                box.x1 = ret1[i+2]+ 1920;   //扩大框范围  0 像素
                box.y1 = ret1[i+3];
                box.x2 = ret1[i+4]+ 1920;
                box.y2 = ret1[i+5];
                boxes_xxyy.push_back(box);
            }
        }
        for (int i=0; i< ret2.size(); i+=6)
        {
            if (ret2[i] ==mode)    //head
            {
                box.x1 = ret2[i+2];   //hard code
                box.y1 = ret2[i+3];
                box.x2 = ret2[i+4];
                box.y2 = ret2[i+5];
                boxes_xxyy.push_back(box);
            }
        }

        for (int i=0; i< ret3.size(); i+=6)
        {
            if (ret3[i] ==mode)    //head
            {
                box.x1 = ret3[i+2] + 960;
                box.y1 = ret3[i+3];
                box.x2 = ret3[i+4] + 960;   //扩大框范围  0 像素
                box.y2 = ret3[i+5];
                boxes_xxyy.push_back(box);
            }
        }
        //change boxes


        head_tracker->run(boxes_xxyy);
        ids = head_tracker->tracking_result;
    }
    // with id
    else{
        Box box;
        for (int i=0; i< ret1.size(); i+=6)
        {
            box.x1 = ret1[i+2]+ 1920;   //扩大框范围  0 像素
            box.y1 = ret1[i+3];
            box.x2 = ret1[i+4]+ 1920;
            box.y2 = ret1[i+5];
            boxes_xxyy.push_back(box);
            ids.push_back(ret1[i]+100);
        }
        for (int i=0; i< ret2.size(); i+=6)
        {
            box.x1 = ret2[i+2];   //hard code
            box.y1 = ret2[i+3];
            box.x2 = ret2[i+4];
            box.y2 = ret2[i+5];
            boxes_xxyy.push_back(box);
            ids.push_back(ret2[i]+200);
        }

        for (int i=0; i< ret3.size(); i+=6)
        {

            box.x1 = ret3[i+2] + 960;
            box.y1 = ret3[i+3];
            box.x2 = ret3[i+4] + 960;   //扩大框范围  0 像素
            box.y2 = ret3[i+5];
            boxes_xxyy.push_back(box);
            ids.push_back(ret3[i]+300);
        }
    }
    //reid
//    std::cout<<"match_id.cpp : "<< Person_id->person_id.size()<<std::endl;
//    for(int i =0;i <Person_id->person_id.size();i++){
//        std::cout<<Person_id->person_id[i]<<std::endl;
//        std::cout<<Person_id->timestamp[i]<<std::endl;
//    }
//    std::cout<<"\n\n\n\n"<< Person_id.person_id.size()<<std::endl;
    // id map
    for (int i =0;i<boxes_xxyy.size();i++)
    {
        //2470 0 2570 140 xyxy
        if (boxes_xxyy[i].x1>2470 && boxes_xxyy[i].x2<2570 && boxes_xxyy[i].y2<140){   // location in
            std::cout<<"location in"<<std::endl;
            if(std::find(match_tid.begin(), match_tid.end(), ids[i])== match_tid.end()){  // first in
                //for test
                std::cout<<"first in in"<<std::endl;
                if (Person_id->person_id.size()>0){                                       // has ready id
                    std::cout<<"has ready id"<<std::endl;
                    std::cout<<Person_id->person_id[0]<<std::endl;
                    std::cout<<Person_id->timestamp[0]<<std::endl;

                    match_tid.push_back(ids[i]);
                    match_pid.push_back(Person_id->person_id[i]);
                    std::cout<<"get person_id"<<" track id="<<ids[i]<<"person id="<<Person_id->person_id[i]<<std::endl;
                    //  del first item
                    vector<int>::iterator k = Person_id->person_id.begin();
                    Person_id->person_id.erase(k);
                    vector<long long>::iterator g = Person_id->timestamp.begin();
                    Person_id->timestamp.erase(g);
                }

            }
        }
    }

    trajectory->update(ids,boxes_xxyy, num);
//    trajectory->print_self();
    cv::Mat frame = img;

    for (int i = 0;i <boxes_xxyy.size(); i++){

        int new_id = trajectory->get_merged_id(ids[i]);

        vector<int> draw_list = trajectory->get_draw_list(new_id);
//        std::cout<<" new id:"<<new_id<< "len draw_list"<< draw_list.size()<<std::endl;
//        std::string text = std::to_string(ids[i]);

        std::string text = std::to_string(new_id) + "-" + to_string(find_pid(new_id));

        cv::rectangle(frame,cv::Point(boxes_xxyy[i].x1,boxes_xxyy[i].y1), cv::Point(boxes_xxyy[i].x2,boxes_xxyy[i].y2),cv::Scalar(0,255,0),1,1,0);
        cv::putText(frame, text, cv::Point(boxes_xxyy[i].x1,boxes_xxyy[i].y1+100), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0,0,255),6,9);
        // tracks
        if (draw_list.size()>=4){
            for(int j =0 ;j<(draw_list.size()-4);j+=2){
                if (abs(draw_list[j] - draw_list[j + 2])< 400 &&
                            abs(draw_list[j+1] - draw_list[j+ 3])< 400)
                {
                    cv::line(frame,cv::Point(draw_list[j], draw_list[j+1]),
                           cv::Point(draw_list[j + 2], draw_list[j+ 3]),
                           cv::Scalar(0,255,0),5, 8, 0);

                }

//            std::cout<<" j :"<<j<<endl;
//            std::cout<<" draw :"<<draw_list[j]<<" "<<draw_list[j+1]<<" "<<draw_list[j + 2]<<" "<<draw_list[j+ 3]<<std::endl;
            }
        }

    }
    result_img  = frame;

    std::string img_detections = "./debug/33/" + to_string(num+10000) + ".jpg";
//    std::string img_detections1 = "./debug/33/" + to_string(num+10000) + ".jpg";

//    std::cout<<"write"<<num <<std::endl;
//    cv::imwrite(img_detections, frame);
//    cv::imwrite(img_detections1, ret_img);
}
//
// Created by xinxuehsi  on 2020/5/7.
//
#include <algorithm>
#include "structures/trajectory.h"
#include <cmath>
Trajectory::Trajectory(int dis)
{
    distance = dis;
}

Trajectory::~Trajectory() = default;

pair<int,int> box2center(Box box)
{
    pair<int, int> p1{int((box.x1+box.x2)/2),int((box.y1+box.y2)/2)};
    return p1;
}

bool Trajectory::is_exit_id(int id){
    bool is_exit = false;
    for(int i=0;i<track_ids.size();i++)
    {
        if(track_ids[i]==id)
            is_exit = true;
    }
    return is_exit;
}
int Trajectory::get_order(int id){
    int order;
    for(int i=0;i<track_ids.size();i++)
    {
        if(track_ids[i]==id)
            order = i;
    }
    return order;
}

bool Trajectory::in_merge_arear(int y)
{
    if (Y-100< y <Y+100)
    return true;
    else
    return false;
}


void Trajectory::update_trajectory(vector<int> ids,vector<Box> boxes_xxyy,int frame_count)
{
//    std::cout <<"update"<<std::endl;
//    live_track_ids.swap(vector<int>()); //清空
    live_track_ids.clear();
    for (int i = 0;i <ids.size(); i++){
//        std::cout<<"id :" <<ids[i]<<std::endl;
        // new id
        if (!is_exit_id(ids[i])){
            //new Track_list
            Track_list track_list;
            track_list.track_id = ids[i];
            track_list.start_frame = frame_count;
            track_list.end_frame = frame_count;
            track_list.is_live = true;
            pair<int,int> t1;
            track_list.traject.push_back(box2center(boxes_xxyy[i]));

            //updata Trajectory
            track_ids.push_back(ids[i]);
            all_track_list.push_back(track_list);

        }

        //live id
        if (is_exit_id(ids[i])){
            int oial = get_order(ids[i]);   //order in all list
            all_track_list[oial].end_frame = frame_count;
            all_track_list[oial].is_live = true;
            all_track_list[oial].traject.push_back(box2center(boxes_xxyy[i]));

            //是否需要合并  放入 列表中
            if (box2center(boxes_xxyy[i]).first<300)  //左端出
            {

            }

        }
    }
}

void Trajectory::merge_trajectory(){
    //边缘的合并
//    for(int i = 0;i <traject.size(); i++){
    //过滤短暂的
//    traject_long
//    }


}

void Trajectory::print_self(){
    std::cout <<"Trajectory: len "<<track_ids.size()<<std::endl;

    for(int i = 0;i <track_ids.size(); i++){

    cout <<"     order"<< i <<" id "<<track_ids[i] <<" " <<all_track_list[i].start_frame<<" --  " <<all_track_list[i].end_frame<< "live: "<<all_track_list[i].is_live
    <<" "<<endl;

    }

}
void Trajectory::judge(int id,Box box)
{
    int center_x = int((box.x1 +box.x2)/2);
    int diff = min (min(abs(center_x - borders), abs(center_x - borders1)), min(abs(center_x - borders2),abs(center_x-0))) ;
    if (diff < distance && std::find(merge_ids.begin(), merge_ids.end(), id)== merge_ids.end())
    {
        merge_ids.push_back(id);
        groups.push_back(merge_ids);
//        std::cout<<"judge:   merge_ids.push_back"<< id <<std::endl;
    }


}

void Trajectory::judge_out(int id,Box box)
{
    int center_x = int((box.x1 +box.x2)/2);
    int diff = min (min(abs(center_x - borders), abs(center_x - borders1)), min(abs(center_x - borders2),abs(center_x-0)));
//    std::cout<<"diff=  "<<diff <<" id="<<id <<std::endl;
    if (diff > distance && std::find(merge_ids.begin(), merge_ids.end(), id)!= merge_ids.end())
    {
        merge_ids.clear();
//        std::cout<<"judge out:   merge_ids.clear" <<std::endl;
    }

}
bool vector_not_int(vector<int> a, vector<vector<int>>b)
{
    bool ret = true;
    for (int i=0;i<b.size();i++)
    {
        if (b[i] == a)
            ret = false;
    }
    return true;

}
bool Trajectory::all_out(vector<int> ids,vector<Box> boxes_xxyy)
{
    bool ret;
    vector<int> diffs;
    for(int i=0;i<boxes_xxyy.size();i++)
    {
        int center_x = (boxes_xxyy[i].x1 + boxes_xxyy[i].x2) / 2;
        int diff = min (min(abs(center_x - borders), abs(center_x - borders1)), min(abs(center_x - borders2),abs(center_x-0)));
        if (diff<distance)
            diffs.push_back(1);
    }

    if (diffs.size() == 0){
        if(vector_not_int(merge_ids,groups)){
            if(merge_ids.size()>0)             // diff with python
                groups.push_back(merge_ids);
        }
        merge_ids.clear();
        ret = true;
    }

    else
    {
        ret=  false;
    }
    return ret;
}

void Trajectory::update(vector<int> ids,vector<Box> boxes_xxyy,int frame_count)
{
    bool out = all_out(ids,boxes_xxyy);
    if (!out){
        for(int i =0; i<boxes_xxyy.size();i++)
        {
            judge(ids[i],boxes_xxyy[i]);
            judge_out(ids[i],boxes_xxyy[i]);
        }
//        std::cout<<"not all_out"<<std::endl;
    }
    else
    merge_group();
    remove_duplicates();
    for (int i=0;i<boxes_xxyy.size();i++){
        int new_id = get_merged_id(ids[i]);
        if (std::find(draw_id.begin(), draw_id.end(), new_id)== draw_id.end())  //不在  新建
        {
            draw_id.push_back(new_id);
            vector<int> tmp;
            tmp.push_back(int((boxes_xxyy[i].x1+boxes_xxyy[i].x2)/2));
            tmp.push_back(int((boxes_xxyy[i].y1+boxes_xxyy[i].y2)/2));
            draw_list.push_back(tmp);
//            std::cout<<"x= "<<int((boxes_xxyy[i].x1+boxes_xxyy[i].x2)/2)<<" y="<<int((boxes_xxyy[i].y1+boxes_xxyy[i].y2)/2)<<std::endl;
        }
        else{                                                                  //添加到原有
            int nPosition;                                                     //原有 ID 的位置
            vector <int>::iterator iElement = find(draw_id.begin(),draw_id.end(),new_id);
            if( iElement != draw_id.end() )
	        {
	            nPosition = std::distance(draw_id.begin(),iElement);
	            draw_list[nPosition].push_back(int((boxes_xxyy[i].x1+boxes_xxyy[i].x2)/2));
                draw_list[nPosition].push_back(int((boxes_xxyy[i].y1+boxes_xxyy[i].y2)/2));
//                std::cout<<"x= "<<int((boxes_xxyy[i].x1+boxes_xxyy[i].x2)/2)<<" y="<<int((boxes_xxyy[i].y1+boxes_xxyy[i].y2)/2)<<std::endl;
	        }

        }
    }
//    std::cout<<"update done"<<std::endl;
//print
//    for(int i=0;i<groups.size();i++)
//    {
//        std::cout<<" group [";
//        for(int j=0;j<groups[i].size();j++)
//        {
//            std::cout<<groups[i][j]<<",";
//        }
//        std::cout<<"]";
//
//    }
//    std::cout<<" merged_id [";
//    for(int i=0;i<merge_ids.size();i++)
//    {    std::cout<<merge_ids[i]<<",";}
//     std::cout<<"]";
}

vector<int> Trajectory::get_draw_list(int new_id)
{
        int nPosition;                                                     //原有 ID 的位置
        vector <int>::iterator iElement = find(draw_id.begin(),draw_id.end(),new_id);
        if( iElement != draw_id.end() )
        {
            nPosition = std::distance(draw_id.begin(),iElement);
            return draw_list[nPosition];
        }
        else{
            std::cout<<"new_id is not in draw_id"<<std::endl;
            vector<int> ret_tmp;
            return ret_tmp;
        }
}


bool has_same_item(const std::vector<int> &nLeft,const std::vector<int> &nRight)
{
	std::vector<int> nResult;
	for (std::vector<int>::const_iterator nIterator = nLeft.begin(); nIterator != nLeft.end(); nIterator++)
	{
		if(std::find(nRight.begin(),nRight.end(),*nIterator) != nRight.end())
			nResult.push_back(*nIterator);
	}
    if (nResult.size()>0)
        return true;
    else
        return false;
}
vector<int> combine_vector(const std::vector<int> &nLeft,const std::vector<int> &nRight)
{
    std::vector<int> nResult;
    nResult.resize(nLeft.size()+nRight.size());
//    sort(nLeft.begin(),nLeft.end());
//	sort(nRight.begin(),nRight.end());
	merge(nLeft.begin(),nLeft.end(),nRight.begin(),nRight.end(),nResult.begin());
	return nResult;
}

void Trajectory::merge_group(){
    if (groups.size()>0){
        vector<int> last_group = groups[groups.size()-1];
        for(int i =0;i<groups.size()-1;i++)
        {
            if(has_same_item(last_group,groups[i]))
            {
                groups[i]= combine_vector(last_group,groups[i]);
                groups.pop_back();
            }
        }
    }
}
int  Trajectory::get_merged_id(int ori_id){

    for(int i=0;i<groups.size();i++){

        if(std::find(groups[i].begin(), groups[i].end(), ori_id)!= groups[i].end())
        return groups[i][0];

    }
    return ori_id;

}
void Trajectory::remove_duplicates(){
    for(int i=0; i<groups.size();i++)
    {
//        groups[i]

    }
}


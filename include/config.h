//
// Created by Orangels on 2020-03-19.
//

#ifndef CONFIG_CONFIG_H
#define CONFIG_CONFIG_H
#include <iostream>
#include <unordered_map>
#include <fstream>

using namespace std;

class Cconfig {

public:
    unordered_map<string, string> labels;

    inline Cconfig(string filePath){
        FILE* fp = fopen(filePath.c_str(), "r");

        while (!feof(fp))
        {
            char str[1024];
            fgets(str, 1024, fp);  //¶ÁÈ¡Ò»ÐÐ
            string str_s(str);

            if (str_s.length() > 0)
            {
                for (int i = 0; i < str_s.length(); i++)
                {
                    if (str_s[i] == ' ')
                    {
                        string strr = str_s.substr(i+1, str_s.length() - i - 1);
                        string key = str_s.substr(0, i);
                        trim(strr);
                        labels[key] = strr;
                        i = str_s.length();
                        break;
                    }
                }
            }
        }
    };
    ~Cconfig(){
        labels.clear();
    };

    void trim(string &s)
    {
        /*
        if( !s.empty() )
        {
            s.erase(0,s.find_first_not_of(" "));
            s.erase(s.find_last_not_of(" ") + 1);
        }
        */
        int index = 0;
        if( !s.empty())
        {
//            while( (index = s.find(' ',index)) != string::npos)
//            {
//                s.erase(index,1);
//            }
            s.erase(s.find_last_not_of("\n") + 1);
        }
    }

    string operator[](string key)
    {
        auto iValue = labels.find(key);
        if (iValue != labels.end())
            return iValue->second;
        else{
            return "";
        }
    }


};

#endif //CONFIG_CONFIG_H

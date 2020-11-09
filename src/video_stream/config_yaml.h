#ifndef YAML_CONFIG_H
#define YAML_CONFIG_H
#include <iostream>
#include "yaml-cpp/yaml.h"
#include "singleton.h"

class yamlConfig {
    friend Singleton<yamlConfig>;

public:
    void reloadConfig(const std::string& path){
        lock_guard<mutex> lock(mtx);
        conf = YAML::LoadFile(path);
    }

    YAML::Node getConfig(){
        return conf;
    }

private:

    yamlConfig() {
        const std::string CONF = "config.yaml";
        conf = YAML::LoadFile(CONF);
    }

    yamlConfig(const std::string& path) {
        conf = YAML::LoadFile(path);
    }

    YAML::Node conf;
    mutex mtx;
};

#endif
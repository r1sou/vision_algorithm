#pragma once

#include "core/model.h"
#include "core/node.h"
#include "utils/client.h"

class Engine
{
public:
    Engine()
    {
        preprocess_pool.init(2);
        postprocess_pool.init(2);

        std::string config_file = CONFIG;
        loadJson(config_file,config);
    }
    ~Engine()
    {
        hbDNNRelease(packed_dnn_handle);
    }

public:
    void InitializeCamera(nlohmann::json config);
    bool InitializeModel(nlohmann::json config, int route = 4);
    bool InitClient(nlohmann::json config, int n_pool = 4);

    void Inference(){
        for(int i = 0;i < node_m->sub_nodes_.size();i++){
            InferenceParallel(i,true);
        }
    }

    void PublishObject(const std::shared_ptr<ModelOutput> &output,int index);
    void PublishLaserScan(const std::shared_ptr<ModelOutput> &output,int index);
    // void PublishLaser(std::shared_ptr<ModelOutput> &output,int index);
    
    // test demo
    void InferenceParallel(int index = 0,bool publish = false);
    void InferenceSerial(int index = 0);

public:
    nlohmann::json config;
    nlohmann::json detect_config;
    std::unordered_set<std::string> need_detect_object_names;

private:
    std::shared_ptr<WebSocketClient> client_object;
    std::shared_ptr<UDPClient> client_laser;

    hbPackedDNNHandle_t packed_dnn_handle;

    ThreadPool preprocess_pool, postprocess_pool;
    ThreadPool publish_pool;

    std::shared_ptr<NodeManage> node_m;
    std::vector<std::shared_ptr<BaseModel>> model_m;
};

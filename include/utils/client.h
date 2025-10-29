#pragma once

#include "common.h"

#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"
#include "websocketpp/client.hpp"

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using Client = websocketpp::client<websocketpp::config::asio_client>;

class WebSocketClient
{
public:
    WebSocketClient(nlohmann::json config, int pool = 2);
    ~WebSocketClient();
    bool Connect(const std::string uri);
    void SendMsg(const std::string message);

private:
    void on_open(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, websocketpp::config::asio_client::message_type::ptr msg);
    void on_close(websocketpp::connection_hdl hdl);

public:
    nlohmann::json m_config;
    std::atomic<bool> m_status;
    Client m_client;
    websocketpp::connection_hdl m_handle;
    websocketpp::lib::shared_ptr<websocketpp::lib::thread> m_thread;
    ThreadPool m_thpool;
};

class JWTGenerator
{
public:
    static std::string generate(std::string req_id, std::string secret);
    static std::string base64url_encode(const unsigned char *data, size_t len);
};

class UDPClient
{
public:
    UDPClient(const std::string ip, uint16_t port);
    ~UDPClient();

public:
    void SendMsg(const std::string message);

public:
    int sockfd_;
    struct sockaddr_in dest_addr_;
    bool valid_;
};
#pragma once
#include "Define.h"
//#include <windows.h>
#include <iostream>

#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define BUFSIZE	128			



class ClientSocket
{
private:
	//server address
    struct sockaddr_in server;

	//client SOCKET that connect to the server
	int clientSocket;

	//the message buffer that is received from the server
	char recvBuf[BUFSIZE];

	//the message buffer that is sent to the server
	char sendBuf[BUFSIZE];

public:
	ClientSocket();
	~ClientSocket(void);

	//connect to the server
	int connectServer();

	//receive message from server
	int recvMsg();

	//send message to server
	int sendMsg(const char* msg);

	//close socket
	void close();


	//get the received message
	char* getRecvMsg();

};


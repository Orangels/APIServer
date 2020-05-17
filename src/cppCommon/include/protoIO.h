#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::Message;

class CProtoIO
{
public:
	CProtoIO() {}
	~CProtoIO() {}
	bool readProtoFromTextFile(const std::string& vFileName, Message* proto)
	{
		int fd = _getFileDes(vFileName.c_str(), true);
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		_closeFile();
		return success;
	}
	void writeProtoToTextFile(const Message& proto, const std::string& vFileName)
	{
		int fd = _getFileDes(vFileName.c_str(), false);
		FileOutputStream* output = new FileOutputStream(fd);
		google::protobuf::TextFormat::Print(proto, output);
		delete output;
		_closeFile();
	}

	bool readProtoFromBinaryFile(const std::string& vFileName, Message* proto)
	{
		const int kProtoReadBytesLimit = (1 << 31) - 1;
		int fd = open(vFileName.c_str(), O_RDONLY);
		if (fd==-1) { std::cout << "Fail to open file  " << vFileName << std::endl; return false; }
		auto* raw_input = new FileInputStream(fd);
		auto* coded_input = new google::protobuf::io::CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

		bool success = proto->ParseFromCodedStream(coded_input);

		delete coded_input;
		delete raw_input;
		close(fd);
		return success;
// 
// 		bool success = proto->ParseFromCodedStream(coded_input);
// 		m_io = std::fstream(vFileName.c_str(), std::ios::in | std::ios::binary);
// 		if (!m_io.is_open()) {std::cout << "Fail to open file  " << vFileName << std::endl; return false;}
// 		bool succes = proto->ParseFromIstream(&m_io);
// 		m_io.close();
// 		return succes;
	}

	bool writeProto2BinaryFile(const std::string& vFileName, Message* proto)
	{
		m_io = std::fstream(vFileName.c_str(), std::ios::out | std::ios::binary);
		if (!m_io.is_open()) { std::cout << "Fail to open file  " << vFileName << std::endl; return false; }
		bool succes = proto->SerializeToOstream(&m_io);
		m_io.close();
		return succes;
	}

private:
	std::fstream m_io;
	int m_fileDes;
	int _getFileDes(const std::string& vFileName, bool vRead)
	{
#ifndef _MBCS
		m_io = std::fstream(vFileName.c_str(), vRead?std::ios::in:std::ios::out);
		auto helper = [](std::filebuf& fb) -> int
		{
			class Helper : public std::filebuf {
			public: int handle() { return _M_file.fd(); }
			};
			return static_cast<Helper&>(fb).handle();
		};
		m_fileDes = helper(*m_io.rdbuf());
#else
		m_fileDes = vRead?open(vFileName.c_str(),  O_RDONLY): open(vFileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
		return m_fileDes;
	}
	void _closeFile()
	{
#ifndef _MBCS
		m_io.close();
#else
		close(m_fileDes);
#endif
	}
};


/*
examples:
#include "caffe.pb.h"
CProtoIO pio;
caffe::NetParameter proto;
pio.readProtoFromBinaryFile("../nameAgeGenderMeanVariance.caffemodel", &proto);
for (int i=0; i<proto.layer_size(); ++i)
{
auto l = proto.mutable_layer(i);
l->clear_blobs();
}
pio.writeProtoToTextFile(proto, "0nameAgeGenderMeanVariance.txt");
*/

//
//  ViewController.m
//  AudioRecognition
//
//  Created by Jeff Tang on 1/23/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVAudioRecorder.h>
#import <AVFoundation/AVAudioSettings.h>
#import <AVFoundation/AVAudioSession.h>
#import <AudioToolbox/AudioToolbox.h>

#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

UIButton *_btn;
UILabel *_lbl;

const int SAMPLE_RATE = 16000;
float *floatInputBuffer;

std::string audioRecognition(float* floatInputBuffer, int length);

@interface ViewController () <AVAudioRecorderDelegate>
@property (nonatomic, strong) NSString *recorderFilePath;
@property (nonatomic, strong) AVAudioRecorder *recorder;
@end


@implementation ViewController
- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    NSLog(@"viewDidLoad: %@", [NSThread currentThread]);
    _btn = [UIButton buttonWithType:UIButtonTypeSystem];
    [_btn setTranslatesAutoresizingMaskIntoConstraints:NO];
    _btn.titleLabel.font = [UIFont systemFontOfSize:32];
    [_btn setTitle:@"Start" forState:UIControlStateNormal];
    [self.view addSubview:_btn];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:_btn
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:_btn
                                                                attribute:NSLayoutAttributeCenterY
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeCenterY
                                                               multiplier:1
                                                                 constant:0];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];

    _lbl = [[UILabel alloc] init];
    [_lbl setTranslatesAutoresizingMaskIntoConstraints:NO];
    [self.view addSubview:_lbl];
    
    NSLayoutConstraint *horizontal2 = [NSLayoutConstraint constraintWithItem:_lbl
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical2 = [NSLayoutConstraint constraintWithItem:_lbl
                                                                attribute:NSLayoutAttributeTop
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeTop
                                                               multiplier:1
                                                                 constant:150];
    [self.view addConstraint:horizontal2];
    [self.view addConstraint:vertical2];

    
    
    [_btn addTarget:self action:@selector(btnTapped:) forControlEvents:UIControlEventTouchUpInside];

    

}


- (IBAction)btnTapped:(id)sender {
    NSLog(@"btnTapped");
    [_btn setTitle:@"Listening..." forState:UIControlStateNormal];
    [_lbl setText:@"..."];

    AVAudioSession *audioSession = [AVAudioSession sharedInstance];
    
    NSError *err = nil;
    [audioSession setCategory:AVAudioSessionCategoryRecord error:&err];
    if(err){
        NSLog(@"audioSession: %@", [[err userInfo] description]);
        return;
    }
    
    [audioSession setActive:YES error:&err];
    if(err){
        NSLog(@"audioSession: %@", [[err userInfo] description]);
        return;
    }
    
    NSMutableDictionary *recordSetting = [[NSMutableDictionary alloc] init];
    [recordSetting setValue:[NSNumber numberWithInt:kAudioFormatLinearPCM] forKey:AVFormatIDKey];
    [recordSetting setValue:[NSNumber numberWithFloat:SAMPLE_RATE] forKey:AVSampleRateKey];
    [recordSetting setValue:[NSNumber numberWithInt: 1] forKey:AVNumberOfChannelsKey];
    [recordSetting setValue :[NSNumber numberWithInt:16] forKey:AVLinearPCMBitDepthKey];
    [recordSetting setValue :[NSNumber numberWithBool:NO] forKey:AVLinearPCMIsBigEndianKey];
    [recordSetting setValue :[NSNumber numberWithBool:NO] forKey:AVLinearPCMIsFloatKey];
    [recordSetting setValue:[NSNumber numberWithInt:AVAudioQualityMax] forKey:AVEncoderAudioQualityKey];
    
    self.recorderFilePath = [NSString stringWithFormat:@"%@/recorded_file.wav", [NSHomeDirectory() stringByAppendingPathComponent:@"tmp"]];
    NSLog(@"recordPause: recorderFilePath=%@", _recorderFilePath);
    NSURL *url = [NSURL fileURLWithPath:_recorderFilePath];
    err = nil;
    _recorder = [[ AVAudioRecorder alloc] initWithURL:url settings:recordSetting error:&err];
    if(!_recorder){
        NSLog(@"recorder: %@", [[err userInfo] description]);
        return;
    }
    //prepare to record
    [_recorder setDelegate:self];
    [_recorder prepareToRecord];
    [_recorder recordForDuration:1];
}

- (void)audioRecorderDidFinishRecording:(AVAudioRecorder *) aRecorder successfully:(BOOL)flag
{
    if (!flag) {
        NSLog(@"audioRecorderDidFinishRecording - recording failed");
        return;
    }
    
    const char *cString = [self.recorderFilePath cStringUsingEncoding:NSASCIIStringEncoding];
    
    CFStringRef str = CFStringCreateWithCString(NULL, cString, kCFStringEncodingMacRoman);
    CFURLRef inputFileURL = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, str, kCFURLPOSIXPathStyle, false);
    
    ExtAudioFileRef fileRef;
    ExtAudioFileOpenURL(inputFileURL, &fileRef);
    
    AudioStreamBasicDescription audioFormat;
    audioFormat.mSampleRate = SAMPLE_RATE;
    audioFormat.mFormatID = kAudioFormatLinearPCM;
    audioFormat.mFormatFlags = kLinearPCMFormatFlagIsFloat;
    audioFormat.mBitsPerChannel = sizeof(Float32) * 8;
    audioFormat.mChannelsPerFrame = 1;
    audioFormat.mBytesPerFrame = audioFormat.mChannelsPerFrame * sizeof(Float32);
    audioFormat.mFramesPerPacket = 1;
    audioFormat.mBytesPerPacket = audioFormat.mFramesPerPacket * audioFormat.mBytesPerFrame;
    
    ExtAudioFileSetProperty(fileRef, kExtAudioFileProperty_ClientDataFormat, sizeof (AudioStreamBasicDescription), &audioFormat);
    
    int numSamples = 1024;
    UInt32 sizePerPacket = audioFormat.mBytesPerPacket;
    UInt32 packetsPerBuffer = numSamples;
    UInt32 outputBufferSize = packetsPerBuffer * sizePerPacket;
    
    UInt8 *outputBuffer = (UInt8 *)malloc(sizeof(UInt8 *) * outputBufferSize);
    
    AudioBufferList convertedData ;
    
    convertedData.mNumberBuffers = 1;
    convertedData.mBuffers[0].mNumberChannels = audioFormat.mChannelsPerFrame;
    convertedData.mBuffers[0].mDataByteSize = outputBufferSize;
    convertedData.mBuffers[0].mData = outputBuffer;
    
    UInt32 frameCount = numSamples;
    float *samplesAsCArray;
    int j =0;
    
    int totalRead = 0;
    floatInputBuffer = new float[SAMPLE_RATE*3];
    while (frameCount > 0) {
        ExtAudioFileRead(fileRef, &frameCount, &convertedData);
        //printf("frameCount=%d\n", frameCount);
        totalRead += frameCount;
        if (frameCount > 0)  {
            AudioBuffer audioBuffer = convertedData.mBuffers[0];
            samplesAsCArray = (float *)audioBuffer.mData;
            
            for (int i =0; i<1024; i++) {
                floatInputBuffer[j] = (float)samplesAsCArray[i] ;
                //if (i%20 == 0) printf("%f,",floatInputBuffer[j]);  // -1 TO +1
                j++;
            }
        }
    }
    
    printf("totalRead=%d\n", totalRead);
    
    [_btn setTitle:@"Recognizing..." forState:UIControlStateNormal];

    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        std::string command = audioRecognition(floatInputBuffer, totalRead);
        delete [] floatInputBuffer;
        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *cmd = [NSString stringWithCString:command.c_str() encoding:[NSString defaultCStringEncoding]];
            NSLog(@"cmd=%@", cmd);
            [_lbl setText:cmd];
            [_btn setTitle:@"Start" forState:UIControlStateNormal];
        });
    });
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end


namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                             new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

std::string audioRecognition(float* floatInputBuffer, int length) {
    NSLog(@"audioProcessing: %@", [NSThread currentThread]);
    std::string commands[] = {"_silence_", "_unknown_",
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go"};
    
    tensorflow::SessionOptions options;
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return "";
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    
    NSString* network_path = FilePathForResourceName(@"speech_commands_graph", @"pb");
    
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return "";
    }

    std::string input_name1 = "decoded_sample_data:0";
    std::string input_name2 = "decoded_sample_data:1";
    std::string output_name = "labels_softmax";
    
    tensorflow::Tensor samplerate_tensor(tensorflow::DT_INT32,
                                    tensorflow::TensorShape());
    samplerate_tensor.scalar<int>()() = SAMPLE_RATE;

    tensorflow::Tensor audio_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({length, 1}));
    auto audio_tensor_mapped = audio_tensor.tensor<float, 2>();
    float* out = audio_tensor_mapped.data();
    for (int i = 0; i < length; i++) {
        out[i] = floatInputBuffer[i];
    }
    
    
    std::vector<tensorflow::Tensor> outputScores;
    tensorflow::Status run_status = session->Run({{input_name1, audio_tensor}, {input_name2, samplerate_tensor}},
                                                 {output_name}, {}, &outputScores);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return "";
    }
    tensorflow::string status_string = run_status.ToString();
    tensorflow::Tensor* output = &outputScores[0];
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = output->flat<float>();
    const long count = prediction.size();
    int idx = 0;
    float max = prediction(0);
    for (int i = 1; i < count; i++) {
        const float value = prediction(i);
        printf("%d: %f", i, value);
        if (value > max) {
            max = value;
            idx = i;
        }
    }

    return commands[idx];
}





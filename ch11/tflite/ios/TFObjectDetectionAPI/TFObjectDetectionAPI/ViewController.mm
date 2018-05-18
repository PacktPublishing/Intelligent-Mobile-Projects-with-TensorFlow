//
//  ViewController.m
//  TFObjectDetectionAPI
//
//  Created by Jeff Tang on 1/8/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"
#include <fstream>
#include <queue>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "string_int_label_map.pb.h"

#include "ios_image_load.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;
using namespace std;

//static NSString *image_name = @"pug1.jpg";
static NSString *image_name = @"wowopics.jpg";
//static NSString *image_name = @"image2.jpg";
const int INPUT_SIZE = 416; //608 for yolo; //416 for tiny-yolo-voc or tiny-yolo (coco)

UIImageView *_iv;
UILabel *_lbl;
float _ratio;
float _top;

void RunInferenceOnImage(NSString *model);
NSString* RunYOLOInferenceOnImage(int model);
@interface ViewController ()

@end

@implementation ViewController

-(void)tapped:(UITapGestureRecognizer *)tapGestureRecognizer {
    
    UIAlertAction* ssd_mobilenet = [UIAlertAction actionWithTitle:@"SSD MobileNet v1 Model" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunInferenceOnImage(@"ssd_mobilenet_v1_frozen_inference_graph");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];
    UIAlertAction* faster_rcnn_inception = [UIAlertAction actionWithTitle:@"Faster_RCNN Inception V2" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunInferenceOnImage(@"faster_rcnn_inceptionv2_frozen_inference_graph");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];
    UIAlertAction* faster_rcnn_resnet101 = [UIAlertAction actionWithTitle:@"Faster_RCNN Resnet 101" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunInferenceOnImage(@"faster_rcnn_resnet101_frozen_inference_graph");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];
    UIAlertAction* retrained_ssd_mobilenet = [UIAlertAction actionWithTitle:@"Retrained SSD MobileNet" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunInferenceOnImage(@"retrained_frozen_inference_graph_mobilenet");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];

    UIAlertAction* yolo2_tiny_voc = [UIAlertAction actionWithTitle:@"YOLO2 Tiny VOC" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunYOLOInferenceOnImage(1);
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];
    
    UIAlertAction* yolo2_tiny_coco = [UIAlertAction actionWithTitle:@"YOLO2 Tiny COCO" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            RunYOLOInferenceOnImage(2);
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
            });
        });
    }];
    
    UIAlertAction* none = [UIAlertAction actionWithTitle:@"None" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {}];
    
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Pick a Model" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:ssd_mobilenet];
    [alert addAction:faster_rcnn_inception];
    [alert addAction:faster_rcnn_resnet101];
    [alert addAction:retrained_ssd_mobilenet];
    [alert addAction:yolo2_tiny_voc];
    [alert addAction:yolo2_tiny_coco];
    [alert addAction:none];
    [self presentViewController:alert animated:YES completion:nil];
    
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    _lbl = [[UILabel alloc] init];
    [_lbl setTranslatesAutoresizingMaskIntoConstraints:NO];
    _lbl.text = @"Tap Anywhere";
    [self.view addSubview:_lbl];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:_lbl
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:_lbl
                                                                attribute:NSLayoutAttributeTop
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeTop
                                                               multiplier:1
                                                                 constant:100];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    _iv = [[UIImageView alloc] initWithFrame:self.view.frame];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_iv];
    _iv.image = [UIImage imageNamed:image_name];
    _ratio = _iv.frame.size.width / _iv.image.size.width;
    _top = (_iv.frame.size.height - (_ratio * _iv.image.size.height))/2;
    
    UITapGestureRecognizer *recognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapped:)];
    [self.view addGestureRecognizer:recognizer];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
        return nullptr;
    }
    return file_path;
}


int LoadLablesFile(const string pbtxtFileName, object_detection::protos::StringIntLabelMap *imageLabels)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    int fileDescriptor = open(pbtxtFileName.c_str(), O_RDONLY);
    
    if( fileDescriptor < 0 )
    {
        std::cerr << " Error opening the file " << std::endl;
        return false;
    }
    
    google::protobuf::io::FileInputStream fileInput(fileDescriptor);
    fileInput.SetCloseOnDelete( true );
    
    if (!google::protobuf::TextFormat::Parse(&fileInput, imageLabels)){
        cerr << "Failed to parse address book." << endl;
        return -1;
    }
    return 0;
    
}

string GetDisplayName(const object_detection::protos::StringIntLabelMap* labels, int index)
{
    string displayName = "";
    for (int i = 0; i < labels->item_size(); i++) {
        const object_detection::protos::StringIntLabelMapItem& item = labels->item(i);
        if (index == item.id()) {
            displayName = item.display_name();
            break;
        }
    }
    
    return displayName;
}


Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
    ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

void DrawTopDetections(std::vector<Tensor>& outputs, int image_width, int image_height, BOOL retrained) {
    object_detection::protos::StringIntLabelMap imageLabels;
    LoadLablesFile([FilePathForResourceName(retrained ? @"pet_label_map" : @"mscoco_label_map", @"pbtxt") UTF8String], &imageLabels);

    auto detection_boxes = outputs[0].flat<float>();
    auto detection_scores = outputs[1].flat<float>();
    auto detection_classes = outputs[2].flat<float>();
    auto num_detections = outputs[3].flat<float>()(0);

    LOG(INFO) << "num_detections: " << num_detections << ", detection_scores size: " << detection_scores.size() << ", detection_classes size: " << detection_classes.size() << ", detection_boxes size: " << detection_boxes.size();
    
    for (int i = 0; i < num_detections; i++) {
        const float score = detection_scores(i);
        
        if (score < 0.28) break;
        
        float left = detection_boxes(i * 4 + 1) * image_width;
        float top = detection_boxes(i * 4 + 0) * image_height;
        float right = detection_boxes(i * 4 + 3) * image_width;
        float bottom = detection_boxes((i * 4 + 2)) * image_height;
        
        string displayName = GetDisplayName(&imageLabels, detection_classes(i));
        
        LOG(INFO) << "Detected " << i << ": " << displayName << ", " << score << ", (" << left << ", " << top << ", " << right << ", " << bottom << ")";
        
        dispatch_async(dispatch_get_main_queue(), ^{
            UIView *bbox = [[UIView alloc] initWithFrame:CGRectMake(_ratio*left, _top + _ratio*top, _ratio*(right-left), _ratio*(bottom-top))];
            bbox.backgroundColor = [UIColor clearColor];
            bbox.layer.borderColor = [UIColor greenColor].CGColor;
            bbox.layer.borderWidth = 1;
            [_iv addSubview:bbox];
            
            CATextLayer *textLayer = [CATextLayer layer];
            textLayer.string = [NSString stringWithCString:(displayName + ": " + to_string(score)).c_str() encoding:[NSString defaultCStringEncoding]];
            textLayer.foregroundColor = [UIColor redColor].CGColor;
            textLayer.fontSize = 12;
            textLayer.frame = CGRectMake(_ratio*left, _top + _ratio*top, 100, 25);
            [_iv.layer addSublayer:textLayer];
        });
    }
}

void removeBoxesAndTexts() {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSArray *subviews = [_iv.subviews copy];
        for (UIView *view in subviews) {
            [view removeFromSuperview];
        }
        
        NSArray *sublayers = [_iv.layer.sublayers copy];
        for (CALayer *layer in sublayers) {
            if ([layer isKindOfClass:[CATextLayer class]])
                [layer removeFromSuperlayer];
        }
    });
}

void RunInferenceOnImage(NSString *model) {
    
    removeBoxesAndTexts();

    NSArray *name_ext = [image_name componentsSeparatedByString:@"."];
    NSString* image_path = FilePathForResourceName(name_ext[0], name_ext[1]);
    string graph = [FilePathForResourceName(model, @"pb") UTF8String];
    
    std::unique_ptr<tensorflow::Session> session;
    Status load_graph_status = LoadGraph(graph, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return;
    }
    
    int image_width;
    int image_height;
    int image_channels;
    
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    
    const int wanted_channels = 3;
    assert(image_channels >= wanted_channels);
    
    tensorflow::Tensor image_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, image_height, image_width, wanted_channels}));
    
    auto image_tensor_mapped = image_tensor.tensor<uint8, 4>();
    tensorflow::uint8* in = image_data.data();
    uint8* c_out = image_tensor_mapped.data();
    for (int y = 0; y < image_height; ++y) {
        tensorflow::uint8* in_row = in + (y * image_width * image_channels);
        uint8* out_row = c_out + (y * image_width * wanted_channels);
        for (int x = 0; x < image_width; ++x) {
            tensorflow::uint8* in_pixel = in_row + (x * image_channels);
            uint8* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = in_pixel[c];
            }
        }
    }
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{"image_tensor", image_tensor}},
                 {"detection_boxes", "detection_scores", "detection_classes", "num_detections"}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return;
    }
    BOOL retrained = [model isEqualToString:@"retrained_frozen_inference_graph_mobilenet"] ? true : false;
    DrawTopDetections(outputs, image_width, image_height, retrained);

    return;
}

//////////////// YOLO2 ///////////


CGSize imgSize;
const char* LABELS_VOC[] = {
    // for tiny-yolo-voc.pb:
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

const char* LABELS_COCO[] = {
    // for yolo.pb and tiny-yolo.pb (coco):
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};
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
}  // namespace


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


float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float vals[], int count) {
    float max = -FLT_MAX;
    for (int i=0; i<count; i++) {
        max = fmax(max, vals[i]);
    }
    float sum = 0.0;
    for (int i=0; i<count; i++) {
        vals[i] = exp(vals[i] - max);
        sum += vals[i];
    }
    for (int i=0; i<count; i++) {
        vals[i] /= sum;
    }
}

static void YoloPostProcess(int model, const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                            Eigen::Aligned>& output, std::vector<std::pair<float, int> >* top_results) {

    const int NUM_CLASSES_COCO = 80; //20 - for tiny-yolo-voc; //80 - for yolo and tiny-yolo coco;
    const int NUM_CLASSES_VOC = 20; //20 - for tiny-yolo-voc; //80 - for yolo and tiny-yolo coco;
    const int NUM_BOXES_PER_BLOCK = 5;
    double ANCHORS_COCO[] = {
        // for tiny-yolo.pb (coco): 80 classes
        0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741
    };
    
    double ANCHORS_VOC[] = {
        // for tiny-yolo-voc.pb: 20 classes
        1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52
    };

    int NUM_CLASSES = model == 1 ? NUM_CLASSES_VOC : NUM_CLASSES_COCO;
    double ANCHORS[10];
    std::copy(std::begin(model == 1 ? ANCHORS_VOC : ANCHORS_COCO), std::end(model == 1 ? ANCHORS_VOC : ANCHORS_COCO), std::begin(ANCHORS));
    
    // 13 for tiny-yolo-voc or tiny-yolo (coco), 19 for yolo (coco)
    const int gridHeight = 13;
    const int gridWidth = 13;
    const int blockSize = 32;
    
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_result_pq;
    
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_rect_pq;
    
    NSMutableDictionary *idxRect = [NSMutableDictionary dictionary];
    NSMutableDictionary *idxDetectedClass = [NSMutableDictionary dictionary];
    int i=0;
    for (int y = 0; y < gridHeight; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
            for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                int offset = (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                + (NUM_CLASSES + 5) * b;
                
                // implementation based on the TF Android TFYoloDetector.java
                // also in http://machinethink.net/blog/object-detection-with-yolo/
                float xPos = (x + sigmoid(output(offset + 0))) * blockSize;
                float yPos = (y + sigmoid(output(offset + 1))) * blockSize;
                
                float w = (float) (exp(output(offset + 2)) * ANCHORS[2 * b + 0]) * blockSize;
                float h = (float) (exp(output(offset + 3)) * ANCHORS[2 * b + 1]) * blockSize;
                
                // Now xPos and yPos represent the center of the bounding box in the 416×416 image that we used as input to the neural network; w and h are the width and height of the box in that same image space.
                CGRect rect = CGRectMake(
                                         fmax(0, (xPos - w / 2) * imgSize.width / INPUT_SIZE),
                                         fmax(0, (yPos - h / 2) * imgSize.height / INPUT_SIZE),
                                         w* imgSize.width / INPUT_SIZE, h* imgSize.height / INPUT_SIZE);
                
                float confidence = sigmoid(output(offset + 4));
                
                float classes[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    classes[c] = output(offset + 5 + c);
                }
                softmax(classes, NUM_CLASSES);
                
                int detectedClass = -1;
                float maxClass = 0;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    if (classes[c] > maxClass) {
                        detectedClass = c;
                        maxClass = classes[c];
                    }
                }
                
                float confidenceInClass = maxClass * confidence;
                if (confidenceInClass > 0.15) {
//                    NSLog(@"%s (%d) %f %d, %d, %d, %@", LABELS[detectedClass], detectedClass, confidenceInClass, y, x, b, NSStringFromCGRect(rect));
                    top_result_pq.push(std::pair<float, int>(confidenceInClass, detectedClass));
                    top_rect_pq.push(std::pair<float, int>(confidenceInClass, i));
                    [idxRect setObject:NSStringFromCGRect(rect) forKey:[NSNumber numberWithInt:i]];
                    [idxDetectedClass setObject:[NSNumber numberWithInt:detectedClass] forKey:[NSNumber numberWithInt:i++]];
                }
            }
        }
    }
    
    
    std::vector<std::pair<float, int> > top_rects;
    while (!top_rect_pq.empty()) {
        top_rects.push_back(top_rect_pq.top());
        top_rect_pq.pop();
    }
    std::reverse(top_rects.begin(), top_rects.end());
    
    
    // Start with the box that has the highest score.
    // Remove any remaining boxes - with the same class? - that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    std::vector<std::pair<float, int> > nms_rects;
    while (!top_rects.empty()) {
        auto& first = top_rects.front();
        CGRect rect_first = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:first.second]]);
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:first.second]] intValue];
        //NSLog(@"first class: %s", LABELS[detectedClass]);
        
        for (unsigned long i = top_rects.size()-1; i>=1; i--) {
            auto& item = top_rects.at(i);
            int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:item.second]] intValue];
            
            CGRect rect_item = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:item.second]]);
            CGRect rectIntersection = CGRectIntersection(rect_first, rect_item);
            if (CGRectIsNull(rectIntersection)) {
                //NSLog(@"no intesection");
                //NSLog(@"no intesection - class: %s", LABELS[detectedClass]);
            }
            else {
                float areai = rect_first.size.width * rect_first.size.height;
                float ratio = rectIntersection.size.width * rectIntersection.size.height / areai;
                //NSLog(@"found intesection - class: %s", LABELS[detectedClass]);
                
                if (ratio > 0.23) {
                    top_rects.erase(top_rects.begin() + i);
                }
            }
        }
        nms_rects.push_back(first);
        top_rects.erase(top_rects.begin());
    }
    
    NSLog(@"nms_rects size=%lu", nms_rects.size());
    while (!nms_rects.empty()) {
        dispatch_async(dispatch_get_main_queue(), ^{
            auto& front = nms_rects.front();
            int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
            //NSLog(@"%f: %s %d %@", front.first, LABELS[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
            
            CGRect rect = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:front.second]]);

            UIView *bbox = [[UIView alloc] initWithFrame:CGRectMake(_ratio*rect.origin.x, _top + _ratio*rect.origin.y, _ratio*(rect.size.width), _ratio*(rect.size.height))];
            bbox.backgroundColor = [UIColor clearColor];
            bbox.layer.borderColor = [UIColor greenColor].CGColor;
            bbox.layer.borderWidth = 1;
            [_iv addSubview:bbox];
            
            CATextLayer *textLayer = [CATextLayer layer];
            textLayer.string = [NSString stringWithFormat:@"%s: %f", (model==1?LABELS_VOC[detectedClass]:LABELS_COCO[detectedClass]), front.first];
            textLayer.foregroundColor = [UIColor redColor].CGColor;
            textLayer.fontSize = 12;
            textLayer.frame = CGRectMake(_ratio*rect.origin.x, _top + _ratio*rect.origin.y, 100, 25);
            [_iv.layer addSublayer:textLayer];
        });
        nms_rects.erase(nms_rects.begin());
    }
    
//    while (!nms_rects.empty()) {
//        auto& front = nms_rects.front();
//        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
//        top_results->push_back(std::pair<float, int>(front.first, detectedClass));
//        
//        NSLog(@"%f: %s %d %@", front.first, LABELS[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
//        nms_rects.erase(nms_rects.begin());
//    }
    
}

NSString* RunYOLOInferenceOnImage(int model) {
    removeBoxesAndTexts();
    
    tensorflow::SessionOptions options;
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return [NSString stringWithFormat: @"Session create failed - %s",
                status_string.c_str()];
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    
    NSString* network_path = FilePathForResourceName(model == 1 ? @"quantized_tiny-yolo-voc" : @"quantized_tiny-yolo", @"pb");
    
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return @"";
    }
    
    NSString* result = [[network_path lastPathComponent] stringByAppendingString: @" - loaded!"];
    
    NSArray *name_ext = [image_name componentsSeparatedByString:@"."];
    NSString* image_path = FilePathForResourceName(name_ext[0], @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = INPUT_SIZE; //416;
    const int wanted_height = INPUT_SIZE; //416;
    const int wanted_channels = 3;
    
    // YOLO’s convolutional layers downsample the image by a factor of 32 so by using an input image of 416 we get an output feature map of 13x13.
    
    assert(image_channels >= wanted_channels);
    
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                //out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                out_pixel[c] = in_pixel[c] / 255.0f; // in Android's TensorFlowYoloDetector.java, no std and mean is used for input values - "We also need to scale the pixel values from integers that are between 0 and 255 to the floating point values that the graph operates on. We control the scaling with the input_mean and input_std flags: we first subtract input_mean from each pixel value, then divide it by input_std." https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
            }
        }
    }
    
    
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{"input", image_tensor}},
                                                 {"output"}, {}, &outputs);
    
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        result = @"Error running model";
        return result;
    }
    tensorflow::string status_string = run_status.ToString();
    result = [NSString stringWithFormat: @"%@ - %s", result, status_string.c_str()];
    
    tensorflow::Tensor* output = &outputs[0];
    std::vector<std::pair<float, int> > top_results;
    
    imgSize = [UIImage imageNamed:image_name].size;

    YoloPostProcess(model, output->flat<float>(), &top_results);
    
//    for (const auto& r : top_results) {
//        const float confidence = r.first;
//        const int index = r.second;
//        result = [NSString stringWithFormat: @"%@\n%f: %s", result, confidence, LABELS[index]];
//        std::cout << confidence << ": " << LABELS[index] << "\n";
//    }
    
    
    return result;
}


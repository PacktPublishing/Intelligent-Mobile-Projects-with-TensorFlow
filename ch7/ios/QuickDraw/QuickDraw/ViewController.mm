//
//  ViewController.m
//  QuickDraw
//
//  Created by Jeff Tang on 2/15/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <queue>
#include <fstream>
#include "tensorflow_utils.h"
using namespace std;

UIImageView *_iv;
UIButton *_btn;
UILabel *_lbl;
BOOL _canDraw = NO;
BOOL _modelLoaded = NO;
const int CLASS_COUNT = 345;

static BOOL USEMEMMAPPED = NO;
static NSString* MODEL_FILE = @"quickdraw_frozen_long_blacklist_strip_transformed";
static NSString* MODEL_FILE_MEMMAPPED = @"quickdraw_frozen_long_blacklist_strip_transformed_memmapped";
static NSString* MODEL_FILE_TYPE = @"pb";

unique_ptr<tensorflow::Session> tf_session;
unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;

std::string getDrawingClassification(NSMutableArray *allPoints);
void normalizeScreenCoordinates(NSMutableArray *allPoints, float *normalized);


@interface ViewController () {
    NSMutableArray *_allPoints;
    NSMutableArray *_consecutivePoints;
}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    _btn = [UIButton buttonWithType:UIButtonTypeSystem];
    [_btn setTranslatesAutoresizingMaskIntoConstraints:NO];
    _btn.titleLabel.font = [UIFont systemFontOfSize:32];
    [_btn setTitle:@"Start" forState:UIControlStateNormal];
    [self.view addSubview:_btn];
    _allPoints = [NSMutableArray array];
    _consecutivePoints = [NSMutableArray array];
    
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
                                                                attribute:NSLayoutAttributeBottom
                                                               multiplier:1
                                                                 constant:-50];
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
                                                                  constant:50];
    [self.view addConstraint:horizontal2];
    [self.view addConstraint:vertical2];

    
    [_btn addTarget:self action:@selector(btnTapped:) forControlEvents:UIControlEventTouchUpInside];
    
    
    _iv = [[UIImageView alloc] init];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [_iv setTranslatesAutoresizingMaskIntoConstraints:NO];
    [self.view addSubview:_iv];
    
    NSLayoutConstraint *horizontal3 = [NSLayoutConstraint constraintWithItem:_iv
                                                                   attribute:NSLayoutAttributeCenterX
                                                                   relatedBy:NSLayoutRelationEqual
                                                                      toItem:self.view
                                                                   attribute:NSLayoutAttributeCenterX
                                                                  multiplier:1
                                                                    constant:0];
    NSLayoutConstraint *vertical3 = [NSLayoutConstraint constraintWithItem:_iv
                                                                 attribute:NSLayoutAttributeTop
                                                                 relatedBy:NSLayoutRelationEqual
                                                                    toItem:self.view
                                                                 attribute:NSLayoutAttributeTop
                                                                multiplier:1
                                                                  constant:0];
    NSLayoutConstraint *width = [NSLayoutConstraint constraintWithItem:_iv
                                                                 attribute:NSLayoutAttributeWidth
                                                                 relatedBy:NSLayoutRelationEqual
                                                                    toItem:self.view
                                                                 attribute:NSLayoutAttributeWidth
                                                                multiplier:1
                                                                  constant:0];
    NSLayoutConstraint *height = [NSLayoutConstraint constraintWithItem:_iv
                                                                 attribute:NSLayoutAttributeHeight
                                                                 relatedBy:NSLayoutRelationEqual
                                                                    toItem:self.view
                                                                 attribute:NSLayoutAttributeHeight
                                                                multiplier:1
                                                                  constant:0];
    [self.view addConstraint:horizontal3];
    [self.view addConstraint:vertical3];
    [self.view addConstraint:width];
    [self.view addConstraint:height];
    
    
}

- (IBAction)btnTapped:(id)sender {
    NSLog(@"btnTapped");
    _canDraw = YES;
    [_btn setTitle:@"Restart" forState:UIControlStateNormal];
    [_lbl setText:@""];
    _iv.image = [UIImage imageNamed:@""];
    [_allPoints removeAllObjects];

//    dispatch_async(dispatch_get_global_queue(0, 0), ^{
//        std::string classes = getDrawingClassification(_allPoints);
//        dispatch_async(dispatch_get_main_queue(), ^{
//            NSString *c = [NSString stringWithCString:classes.c_str() encoding:[NSString defaultCStringEncoding]];
//            NSLog(@"c=%@", c);
//            [_lbl setText:c];
//            //[_btn setTitle:@"Start" forState:UIControlStateNormal];
//        });
//    });
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (UIImage *)createDrawingImageInRect:(CGRect)rect
{
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(rect.size.width, rect.size.height), NO, 0.0);
    UIBezierPath *path = [UIBezierPath bezierPath];
    
    for (NSArray *cp in _allPoints) {
        bool firstPoint = TRUE;
        for (NSValue *pointVal in cp) {
            CGPoint point = pointVal.CGPointValue;
            if (firstPoint) {
                [path moveToPoint:point];
                firstPoint = FALSE;
            }
            else
                [path addLineToPoint:point];
        }
    }
    
    bool firstPoint = TRUE;
    for (NSValue *pointVal in _consecutivePoints) {
        CGPoint point = pointVal.CGPointValue;
        if (firstPoint) {
            [path moveToPoint:point];
            firstPoint = FALSE;
        }
        else
            [path addLineToPoint:point];
    }
    
    path.lineWidth = 6.0;
    [[UIColor blackColor] setStroke];
    [path stroke];
    
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}


- (void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
    if (!_canDraw) return;
    
    [_consecutivePoints removeAllObjects];
    UITouch *touch = [touches anyObject];
    CGPoint point = [touch locationInView:self.view];
    [_consecutivePoints addObject:[NSValue valueWithCGPoint:point]];
    
    _iv.image = [self createDrawingImageInRect:_iv.frame];
    
    
}


- (void) touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
    if (!_canDraw) return;

    UITouch *touch = [touches anyObject];
    CGPoint point = [touch locationInView:self.view];

    //NSLog(@"moved: %f, %f", point.x, point.y);
    [_consecutivePoints addObject:[NSValue valueWithCGPoint:point]];
    _iv.image = [self createDrawingImageInRect:_iv.frame];


}

- (void) touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    if (!_canDraw) return;
    
    UITouch *touch = [touches anyObject];
    CGPoint point = [touch locationInView:self.view];
    [_consecutivePoints addObject:[NSValue valueWithCGPoint:point]];
    [_allPoints addObject:[NSArray arrayWithArray:_consecutivePoints]];
    [_consecutivePoints removeAllObjects];
    
    _iv.image = [self createDrawingImageInRect:_iv.frame];
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        std::string classes = getDrawingClassification(_allPoints);
        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *c = [NSString stringWithCString:classes.c_str() encoding:[NSString defaultCStringEncoding]];
            NSLog(@"c=%@", c);
            [_lbl setText:c];
        });
    });
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

static void GetTopN(const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                    Eigen::Aligned>& prediction,
                    const int num_results, const float threshold,
                    std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>,
    std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > > top_result_pq;
    
    float sum = 0.0;
    for (int i = 0; i < CLASS_COUNT; ++i) {
        sum += expf(prediction(i));
    }
    
    for (int i = 0; i < CLASS_COUNT; ++i) {
        const float value = expf(prediction(i)) / sum;
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}


void normalizeScreenCoordinates(NSMutableArray *allPoints, float *normalized) {
    float lowerx=MAXFLOAT, lowery=MAXFLOAT, upperx=-MAXFLOAT, uppery=-MAXFLOAT;
    for (NSArray *cp in allPoints) {
        for (NSValue *pointVal in cp) {
            CGPoint point = pointVal.CGPointValue;
            if (point.x < lowerx) lowerx = point.x;
            if (point.y < lowery) lowery = point.y;
            if (point.x > upperx) upperx = point.x;
            if (point.y > uppery) uppery = point.y;
        }
    }
    NSLog(@"lowerx=%f, lowery=%f, upperx=%f, uppery=%f", lowerx, lowery, upperx, uppery);
    float scalex = upperx - lowerx;
    float scaley = uppery - lowery;
    
    int n = 0;
    for (NSArray *cp in allPoints) {
        int m = 0;
        for (NSValue *pointVal in cp) {
            CGPoint point = pointVal.CGPointValue;
            normalized[n*3] = (point.x - lowerx) / scalex;
            normalized[n*3+1] = (point.y - lowery) / scaley;
            normalized[n*3+2] = (m ==cp.count-1 ? 1 : 0);
            n++;
            m++;
        }
    }
    
    for (int i=0; i<n-1; i++) {
        normalized[i*3] = normalized[(i+1)*3] - normalized[i*3];
        normalized[i*3+1] = normalized[(i+1)*3+1] - normalized[i*3+1];
        normalized[i*3+2] = normalized[(i+1)*3+2];
    }
}

std::string getDrawingClassification(NSMutableArray *allPoints) {
    if (!_modelLoaded) {
        tensorflow::Status load_status;
        
        if (USEMEMMAPPED) {
            load_status = LoadMemoryMappedModel(MODEL_FILE_MEMMAPPED, MODEL_FILE_TYPE, &tf_session, &tf_memmapped_env);
        }
        else {
            load_status = LoadModel(MODEL_FILE, MODEL_FILE_TYPE, &tf_session);
        }
        
        if (!load_status.ok()) {
            LOG(FATAL) << "Couldn't load model: " << load_status;
            return "";
        }
        _modelLoaded = YES;
    }
    
    
     //cat
    /*float points[] =  {
         130.,   72.,    0.,
         113.,   40.,    0.,
          99.,   27.,    0.,
         109.,   79.,    0.,
          76.,   82.,    0.,
          64.,   88.,    0.,
          55.,  100.,    0.,
          48.,  120.,    0.,
          48.,  134.,    0.,
          51.,  152.,    0.,
          59.,  165.,    0.,
          86.,  184.,    0.,
         133.,  189.,    0.,
         154.,  186.,    0.,
         170.,  179.,    0.,
         203.,  152.,    0.,
         214.,  131.,    0.,
         217.,  114.,    0.,
         215.,  100.,    0.,
         208.,   89.,    0.,
         186.,   76.,    0.,
         176.,    0.,    0.,
         162.,   31.,    0.,
         157.,   65.,    0.,
         132.,   70.,    1.,
          76.,  136.,    0.,
          28.,  128.,    0.,
           7.,  128.,    1.,
          76.,  160.,    0.,
          23.,  164.,    0.,
           0.,  175.,    1.,
          87.,  175.,    0.,
          52.,  191.,    0.,
          37.,  204.,    1.,
         174.,  134.,    0.,
         220.,  132.,    0.,
         246.,  136.,    0.,
         251.,  139.,    1.,
         175.,  147.,    0.,
         255.,  168.,    1.,
         171.,  164.,    0.,
         208.,  198.,    0.,
         215.,  210.,    1.,
         130.,  129.,    0.,
         110.,  134.,    0.,
         108.,  137.,    0.,
         111.,  144.,    0.,
         130.,  148.,    0.,
         139.,  144.,    0.,
         139.,  136.,    0.,
         119.,  130.,    1.,
         107.,   96.,    0.,
        106.,  113.,    1.};

    
    // test: convert points to allPoints
    [allPoints removeAllObjects];
    NSMutableArray *cp = [NSMutableArray array];
    printf("total points: %lu", sizeof(points) / (sizeof(float) * 3));
    for (int i=0; i<sizeof(points) / (sizeof(float) * 3); i++) {
        CGPoint point = CGPointMake(points[i*3], points[i*3+1]);
        [cp addObject:[NSValue valueWithCGPoint:point]];
        if (points[i*3+2] == 1.) {
            [allPoints addObject:[NSArray arrayWithArray:cp]];
            [cp removeAllObjects];
        }
    }*/
//    for (NSArray *cp in allPoints) {
//        printf("********\n");
//        for (NSValue *pointVal in cp) {
//            CGPoint point = pointVal.CGPointValue;
//            printf("%f %f\n", point.x, point.y);
//        }
//    }
    
    
    
    if ([allPoints count] == 0) return "";
    int total_points = 0;
    for (NSArray *cp in allPoints) {
        total_points += cp.count;
    }
    
    // each point after normalization is represented as two floats for points and one additional float indicating whether it's the end of stroke
    float *normalized_points = new float[total_points * 3];
    normalizeScreenCoordinates(allPoints, normalized_points);
    total_points--; // after delta processing, there's one fewer point
//    for (int i=0; i<total_points; i++) {
//        printf("%f, %f, %f\n", normalized_points[i*3], normalized_points[i*3+1], normalized_points[i*3+2]);
//    }
    
    
    
    /*float normalized_points[] =  {
    -0.06666669, -0.15238096,  0.  ,
    -0.05490196, -0.06190476,  0.  ,
     0.03921568,  0.24761905,  0.  ,
    -0.12941176,  0.01428571,  0.  ,
    -0.04705882,  0.02857143,  0.  ,
    -0.03529413,  0.05714285,  0.  ,
    -0.02745098,  0.09523812,  0.  ,
     0.        ,  0.06666666,  0.  ,
     0.01176471,  0.08571428,  0.  ,
     0.03137255,  0.06190473,  0.  ,
     0.10588236,  0.09047621,  0.  ,
     0.18431374,  0.02380949,  0.  ,
     0.08235294, -0.01428568,  0.  ,
     0.06274509, -0.03333336,  0.  ,
     0.12941176, -0.12857139,  0.  ,
     0.04313725, -0.10000002,  0.  ,
     0.01176471, -0.08095235,  0.  ,
    -0.00784314, -0.06666669,  0.  ,
    -0.02745098, -0.05238095,  0.  ,
    -0.0862745 , -0.06190476,  0.  ,
    -0.03921568, -0.36190477,  0.  ,
    -0.05490196,  0.14761905,  0.  ,
    -0.01960784,  0.16190477,  0.  ,
    -0.09803921,  0.02380952,  1.  ,
    -0.21960786,  0.31428573,  0.  ,
    -0.18823531, -0.03809524,  0.  ,
    -0.08235294,  0.        ,  1.  ,
     0.27058825,  0.15238094,  0.  ,
    -0.20784315,  0.01904762,  0.  ,
    -0.09019608,  0.05238092,  1.  ,
     0.34117648,  0.        ,  0.  ,
    -0.13725491,  0.07619047,  0.  ,
    -0.05882353,  0.06190479,  1.  ,
     0.53725493, -0.33333331,  0.  ,
     0.18039215, -0.00952381,  0.  ,
     0.10196078,  0.01904762,  0.  ,
     0.01960784,  0.01428568,  1.  ,
    -0.2980392 ,  0.03809524,  0.  ,
     0.31372547,  0.10000002,  1.  ,
    -0.32941175, -0.01904762,  0.  ,
     0.14509803,  0.16190475,  0.  ,
     0.02745098,  0.05714285,  1.  ,
    -0.33333331, -0.38571429,  0.  ,
    -0.0784314 ,  0.02380955,  0.  ,
    -0.00784314,  0.01428568,  0.  ,
     0.01176471,  0.03333336,  0.  ,
     0.07450983,  0.01904762,  0.  ,
     0.03529412, -0.01904762,  0.  ,
     0.        , -0.03809524,  0.  ,
    -0.0784314 , -0.02857143,  1.  ,
    -0.04705882, -0.16190478,  0.  ,
    -0.00392157,  0.08095238,  1};
    total_points = sizeof(normalized_points) / (sizeof(float) * 3);*/
    
    std::string input_name1 = "Reshape";
    std::string input_name2 = "Squeeze";
    std::string output_name1 = "dense/BiasAdd";
    std::string output_name2 = "ArgMax";
    
    tensorflow::Tensor seqlen_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({8}));
    auto seqlen_mapped = seqlen_tensor.tensor<int64_t, 1>();
    int64_t* seqlen_mapped_data = seqlen_mapped.data();
    for (int i=0; i<8; i++) {
        seqlen_mapped_data[i] = total_points;
        //seqlen_mapped_data[i*2+1] = 3;
    }
    
    tensorflow::Tensor points_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({8, total_points, 3}));
    auto points_tensor_mapped = points_tensor.tensor<float, 3>();
    float* out = points_tensor_mapped.data();
    for (int i=0; i<8; i++) {
        for (int j=0; j<total_points*3; j++)
            out[i*total_points*3+j] = normalized_points[j];
    }

    
    
    std::vector<tensorflow::Tensor> outputs;
    
    tensorflow::Status run_status = tf_session->Run({{input_name1, points_tensor}, {input_name2, seqlen_tensor}},
                                                    {output_name1, output_name2}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Getting model failed:" << run_status;
        return "";
    }

    tensorflow::string status_string = run_status.ToString();
    tensorflow::Tensor* logits_tensor = &outputs[0];
    
    
    NSString* classes_file_path = FilePathForResourceName(@"classes", @"txt");
    if (!classes_file_path) {
        LOG(FATAL) << "Couldn't load classes file: " << classes_file_path;
    }
    ifstream t;
    std::vector<std::string> classes;
    t.open([classes_file_path UTF8String]);
    string line;
    while(t){
        getline(t, line);
        classes.push_back(line);
    }
    t.close();
    
    const int kNumResults = 5;
    const float kThreshold = 0.05f;
    std::vector<std::pair<float, int> > top_results;
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& logits = logits_tensor->flat<float>();

    GetTopN(logits, kNumResults, kThreshold, &top_results);
    string result = "";
    for (int i=0; i<top_results.size(); i++) {
        std::pair<float, int> r = top_results[i];
        printf("%d: %f, %s\n", r.second, r.first, classes[r.second].c_str());
        if (result == "")
            result = classes[r.second]; // + " (" + to_string(r.first) + ") ";
        else result += ", " + classes[r.second];
    }
    
    //long count = logits.size();
    //printf("logits size=%ld, %ld/8=%ld\n\n", count, count, count/8); // count/8=345, number of classes in eval.tfrecord.classes!!!

    
    // no need for this
//    tensorflow::Tensor* argmax_tensor = &outputs[1];
//    const Eigen::TensorMap<Eigen::Tensor<int64_t, 1, Eigen::RowMajor>, Eigen::Aligned>& argmax = argmax_tensor->flat<int64_t>();
//    count = argmax.size();
//    printf("\n\nargmax size=%ld\n", count);
//    for (int i = 0; i < count; i++) {
//        const int64_t value = argmax(i);
//        printf("%lld\n", value);
//    }
    
    //delete []normalized_points;
    
    return result;
}


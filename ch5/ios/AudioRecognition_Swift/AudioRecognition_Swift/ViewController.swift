//
//  ViewController.swift
//  AudioRecognition_Swift
//
//  Created by Jeff Tang on 1/25/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

import UIKit
import AVFoundation

let _lbl = UILabel()
let _btn = UIButton(type: .system)
var _recorderFilePath: String!

extension String {
    func stringByAppendingPathComponent(path: String) -> String {
        let s = self as NSString
        return s.appendingPathComponent(path)
    }
}

class ViewController: UIViewController, AVAudioRecorderDelegate  {
    var audioRecorder: AVAudioRecorder!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        _btn.translatesAutoresizingMaskIntoConstraints = false
        _btn.titleLabel?.font = UIFont.systemFont(ofSize:32)
        _btn.setTitle("Start", for: .normal)
        self.view.addSubview(_btn)
        
        let horizontal = NSLayoutConstraint(item: _btn, attribute: .centerX, relatedBy: .equal, toItem: self.view, attribute: .centerX, multiplier: 1, constant: 0)
        
        let vertical = NSLayoutConstraint(item: _btn, attribute: .centerY, relatedBy: .equal, toItem: self.view, attribute: .centerY, multiplier: 1, constant: 0)

        self.view.addConstraint(horizontal)
        self.view.addConstraint(vertical)
        
        _btn.addTarget(self, action:#selector(btnTapped), for: .touchUpInside)
        
        _lbl.translatesAutoresizingMaskIntoConstraints = false
        self.view.addSubview(_lbl)
        
        let horizontal2 = NSLayoutConstraint(item: _lbl, attribute: .centerX, relatedBy: .equal, toItem: self.view, attribute: .centerX, multiplier: 1, constant: 0)
        
        let vertical2 = NSLayoutConstraint(item: _lbl, attribute: .centerY, relatedBy: .equal, toItem: self.view, attribute: .top, multiplier: 1, constant: 150)
        
        self.view.addConstraint(horizontal2)
        self.view.addConstraint(vertical2)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    
    @objc func btnTapped() {
        _lbl.text = "..."
        _btn.setTitle("Listening...", for: .normal)
        
        AVAudioSession.sharedInstance().requestRecordPermission () {
            [unowned self] allowed in
            if allowed {
                print("mic allowed")
            } else {
                print("denied by user")
                return
            }
        }
        
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(AVAudioSessionCategoryRecord)
            try audioSession.setActive(true)
        } catch {
            print("recording exception")
            return
        }

        let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ] as [String : Any]
        
        do {
            _recorderFilePath = NSHomeDirectory().stringByAppendingPathComponent(path: "tmp").stringByAppendingPathComponent(path: "recorded_file.wav") // can't use m4a extension or error below
            print("recorderFilePath="+_recorderFilePath.description)
            audioRecorder = try AVAudioRecorder(url: NSURL.fileURL(withPath: _recorderFilePath), settings: settings)
            audioRecorder.delegate = self
            audioRecorder.record(forDuration: 1)
        } catch let error {
            print("error:" + error.localizedDescription)
            // error:The operation couldn’t be completed. (OSStatus error 1718449215.)
            // NSOSStatusErrorDomain Code=1718449215 - kAudioFormatUnsupportedDataFormatError
        }
    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        _btn.setTitle("Recognizing...", for: .normal)
        if flag {
            let result = RunInference_Wrapper().run_inference_wrapper(_recorderFilePath)
            _lbl.text = result
        }
        else {
            _lbl.text = "Recording error"
        }
        _btn.setTitle("Start", for: .normal)
    }

}



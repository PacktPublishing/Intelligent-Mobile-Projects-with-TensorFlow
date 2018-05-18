//
//  ViewController.swift
//  HelloTensorFlow_Swift
//
//  Created by Jeff Tang on 1/4/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        let lbl = UILabel()
        lbl.translatesAutoresizingMaskIntoConstraints = false
        lbl.text = "Tap Anywhere"
        self.view.addSubview(lbl)
        
        let horizontal = NSLayoutConstraint(item: lbl, attribute: .centerX, relatedBy: .equal, toItem: self.view, attribute: .centerX, multiplier: 1, constant: 0)
        
        let vertical = NSLayoutConstraint(item: lbl, attribute: .centerY, relatedBy: .equal, toItem: self.view, attribute: .centerY, multiplier: 1, constant: 0)
        
        self.view.addConstraint(horizontal)
        self.view.addConstraint(vertical)
        
        let recognizer = UITapGestureRecognizer(target: self, action: #selector(ViewController.tapped(_:)))
        self.view.addGestureRecognizer(recognizer)
    }
    
    @objc func tapped(_ sender: UITapGestureRecognizer) {
        let alert = UIAlertController(title: "Pick a Model", message: nil, preferredStyle: .actionSheet)
        alert.addAction(UIAlertAction(title: "Inception v3 Retrained Model", style: .default) { action in
            let result = RunInference_Wrapper().run_inference_wrapper("Inceptionv3")
            
            let alert2 = UIAlertController(title: "Inference Result", message: result, preferredStyle: .actionSheet)
            alert2.addAction(UIAlertAction(title: "OK", style: .default) { action2 in
            })
            self.present(alert2, animated: true, completion: nil)
            
        })
        alert.addAction(UIAlertAction(title: "MobileNet 1.0 Retrained Model", style: .default) { action in
            let result = RunInference_Wrapper().run_inference_wrapper("MobileNet")
            
            let alert2 = UIAlertController(title: "Inference Result", message: result, preferredStyle: .actionSheet)
            alert2.addAction(UIAlertAction(title: "OK", style: .default) { action2 in
            })
            self.present(alert2, animated: true, completion: nil)
        })
        alert.addAction(UIAlertAction(title: "None", style: .default) { action in
        })

        self.present(alert, animated: true, completion: nil)
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}


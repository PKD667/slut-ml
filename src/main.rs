#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
#![allow(incomplete_features)]

pub mod plot;
use crate::plot::{plot_comparison,loss_curve};

use slut::{dimension::{self, Dimensionless, *}, dless, dot, tensor::*,units::{self, Unitless}};

const N: usize = 10;
static mut enabled:usize = 3;
static mut threshold:f64 = -5e-5;

fn fac(n: i32) -> i32 {
    let mut r = 1;
    for i in 1..n+1 {
        r*= i;
    }
    r
}

fn infer(coeffs: Vector<f64,Dimensionless,N>, x: f64) -> Scalar<f64,Dimensionless> {
    // create a vector like [1, x¹, x², ... xⁿ⁻¹]
    let mut input_data = [0.0f64; N];
    let r = unsafe{enabled};
    for i in 0..r {
        input_data[i] = x.powi(i as i32);
    }

    let inputs = Vector::<f64,Dimensionless,N>::default(input_data);

    let y = dot!(coeffs, inputs);
    y
}

fn main() {
    println!("{}", fac(3));

    let target = |a: f64| -> f64 {a.cos()};

    let h = dless!(5e-15);
    let mut glr = dless!(1e-4);


    let loss = |r: Scalar<f64,Dimensionless>, t: Scalar<f64,Dimensionless>, coeffs: &Vector<f64,Dimensionless,N>| {
        let d = ((r-t)*(r-t)).mag();
        d
    };

    let mut coeffs = Vector::<f64,Dimensionless,N>::zero();
    println!("{}", coeffs);

    let step = dless!(0.01);
    let max = dless!(5.0);

    // Compute MSE over all training points
    let mse = |c: &Vector<f64,Dimensionless,N>| {
        let mut total_loss = Scalar::<f64,Dimensionless>::zero();
        let num_points = (max / step).raw() as usize;
        
        for i in 0..num_points {
            let x = i as f64 * step.raw();
            total_loss = total_loss + loss(infer(c.clone(), x), dless!(target(x)), c);
        }
        total_loss / dless!(num_points as f64)
    };

    // Gradient using numerical differentiation of MSE
    let grad = |c: &Vector<f64,Dimensionless,N>, k: usize| {
        let mut c_plus = c.clone();
        c_plus.add_at(0, k, 0, h);
        (mse(&c_plus) - mse(c)) / h
    };

    let epochs =  5000;

    println!("Step: {}, Max: {}, Epochs: {}", step, max, epochs);
    println!("Target: {}", target(1.0));

    let starting_loss = mse(&coeffs);
    println!("Starting loss: {}", starting_loss);

    println!("Coeffs: {}", coeffs);

    let mut l = Scalar::zero();
    let mut losses = Vec::<f64>::new();
    let max_gnorm = dless!(10.0);

    let mut last_conv:i32 = -300;

    for e in 0..epochs {
        let mut grads = Vector::<f64,Dimensionless,N>::zero();
        
        // Compute gradient for each coefficient
        for k in 0..N {
            let g = grad(&coeffs, k);
            // Scale down the gradient for higher powers
            let scale = 1.0 / (((k + 1) as f64).powf(1.5) as f64); // or try 1.0 / ((k + 1).pow(2) as f64)
            grads.set_at(0, k, 0, g * dless!(scale));
        }
        
        let g_norm = grads.norm();

        // Gradient clipping
        if g_norm > max_gnorm {
            println!("Gradient norm exceeded threshold, normalizing.");
            grads = grads * (max_gnorm / g_norm);
        }   
        
        coeffs = coeffs - (grads * glr);

        // Compute the loss using the MSE function
        let ln = mse(&coeffs);
        losses.push(l.raw());
        let dl = ln - l;
        l = ln;

        

        // shedule the learning rate decay
        if dl > Scalar::zero() {
            glr = glr * dless!(0.99);
        } else if dl < Scalar::zero()  && e > 50{
            glr = glr * (dless!(1.0)+(dless!(2.0*unsafe{threshold})+dl.mag())*dless!(20.0));
            if glr > dless!(1e-3) {
                glr = dless!(1e-6);
            } else if glr < dless!(1e-6) {
                glr = dless!(1e-6);
            }
        }

        if e % 5 == 0 {
            println!("Gradient: {}", grads);
            println!("Epoch: {}, Loss: {:+e}", e, l.raw());
            println!("Glr: {}, Dl {}", glr.raw(),dl.mag().raw());

            if dl > unsafe {dless!(threshold)} && e - last_conv > 500 {
                println!("Converged at epoch {} with loss {}", e, l.raw());
                last_conv = e as i32;
                unsafe {
                    if enabled < N {
                        enabled += 1;
                    }
                    // scale the threshold inversely with the loss
                    if dl.raw() < unsafe{threshold} && enabled > 1 {
                        threshold = dl.raw() * 2.0;
                    } 
                }
                
            }
        }

        // Save visualizations every 100 epochs
        if e % 100 == 0 && e > 0 {
            let epoch_losses = &losses[0..=e as usize];
            let loss_file = format!("loss_curve.html");
            loss_curve(epoch_losses, &loss_file, Some(unsafe { threshold }))
                .expect("Failed to create loss curve visualization");
            
            let f = |x: f64| infer(coeffs.clone(), x).raw();
            let viz_file = format!("visualization.html");
            plot_comparison(f, target, 0.0, 5.0, 500, &viz_file)
                .expect("Failed to create function comparison visualization");
            
            println!("Saved visualizations for epoch {}", e);
        }

    }

    // Show a sampled version of the losses
    loss_curve(&losses.iter().step_by(1).cloned().collect::<Vec<_>>(), "loss_curve.html", Some(unsafe { threshold }))
        .expect("Failed to create loss curve visualization");

    println!("Final Coeffs: {}", coeffs);
    println!("Final Loss: {}", l);
    println!("Starting Loss: {}", starting_loss);

    let f = |x: f64| infer(coeffs.clone(), x).raw();

    println!("f(1.0) = {}", f(1.0));
    println!("f(1.5) = {}", f(1.5));
    println!("f(2.0) = {}", f(2.0));
    println!("f(3.0) = {}", f(3.0));

    println!("Target(1.0) = {}", target(1.0));
    println!("Target(1.5) = {}", target(1.5));
    println!("Target(2.0) = {}", target(2.0));
    println!("Target(3.0) = {}", target(3.0));

    // Visualization code
    let output_file = "visualization.html";
    plot_comparison(f, target, 0.0, 5.0, 500, output_file)
        .expect("Failed to create visualization");
}

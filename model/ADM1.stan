
data {
  // in unit of response 
  int N; // number of data to fit
  vector<lower=0>[N] sqrt_t;       // sqrt(t)
  real<lower=-1, upper=1> rspn[N]; // response (i.e. position of slider)
  int<lower=1, upper=2> task_n[N]; // task information (1 = perceptual, 2 = value-based)
  
  // in unit of unique trial and stimulus
  int S; // number of unique trial, and stimulus information (num trial * num stim)
  matrix[60,30] stim_x; // stimulus info [trial, stimNum]
  int<lower=1, upper = 60> trialNum[S];   // trial number
  int<lower=1, upper = 30> stimNum[S];    // stimulus number
  int<lower=1> numRspn[S];                //  number of responses for each stimulus
  int<lower=1, upper = 2> task_s[S];      // task (1 = perceptual, 2 = value-based)
  int rspn_start[S];                      // index for start of response 
  int rspn_end[S];                        // index for end of response
  
  // temporal weights
  int W;                     // how long w function is in total? 
  int<lower=1> w_start[29];  // index for start of w for stim num X+1
  int<lower=1> w_end[29];    // index for end of w for stim num X+1
  int ts[W];                // 1:X vector for stim X (for e_p)
  int ts_inv[W];            // X - 1:X + 1 vector for stim X (for e_r)
}

transformed data {
  vector[2] eta; 
  // eta : fix as 0.01
  eta = rep_vector(0.01, 2);
}

parameters {
  vector<lower=0, upper=1>[2] e_p;
  vector<lower=0, upper=1>[2] e_r; 
  vector[2] sigma_w_raw;
}

transformed parameters {
  vector<lower=0>[2] sigma_w; 
  vector[N] mu;
  vector<lower=0>[N] sigma; 
  matrix[W, 2] w;
  
  // transformation of sigma_w
  sigma_w = exp(-1 + sigma_w_raw); // log(sigma_w) ~ N(-1, 1)
  
  // define temporal weight function
  // [weight, task (1 = perceptual, 2 = value-based)]
  for (r in 1:W) {
    for (c in 1:2) {
      w[r, c] = (1-(1-e_p[c]^ts[r])*(1-e_r[c]^ts_inv[r]))*(1-eta[c]) + eta[c];
    }
  }
  
  // define mu and sigma in response unit
  for (i in 1:S) {
    int stimIndex;
    row_vector[stimNum[i]] tmp_w; 
    vector[numRspn[i]] tmp_mu;
    
    stimIndex = stimNum[i] - 1;
    tmp_w = w[w_start[stimIndex]:w_end[stimIndex], task_s[i]]';
    tmp_mu = rep_matrix(tmp_w, numRspn[i]) * stim_x[trialNum[i], 1:stimNum[i]]'/sum(tmp_w);
    
    mu[rspn_start[i]:rspn_end[i]] = tmp_mu;
  }
  
  for (i in 1:N) {
    sigma[i] = sigma_w[task_n[i]]/sqrt_t[i];
  }
}

model {
  // Priors
  e_p ~ beta(2,4); 
  e_r ~ beta(2,4); 
  sigma_w_raw ~ std_normal();
  
  // Likelihood 
  target += normal_lpdf(rspn | mu, sigma);
}



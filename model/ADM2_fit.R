
rm(list=ls()) 

library(dplyr)
library(tidyr)
library(rstan)
options(mc.cores=parallel::detectCores(4))

subj_list = 1:46
subj_exc = unique(c(9, 12, 22, 31, 35, 43,28,45))
subj_list = subj_list[!(subj_list%in%subj_exc)]

# Load data
load("data.RData")


# Load compiled stan model object
# stan_model = stan_model("ADM2.stan")
# saveRDS(stan_model, "ADM2_compiled.RData")
stan_model = readRDS("ADM2_compiled.RData")


for (subjID in subj_list) {
  print(sprintf("running %d", subjID))
  
  # a subject's data to fit
  data = dataAll %>%
    filter(id == subjID) %>%
    group_by(block, trial) %>%
    mutate(sqrt_time = sqrt(1:n() * (1/30))) %>% ungroup()
  
  # extract trial information
  trInfo = data %>%
    group_by(block, task, trial, stimNum) %>%
    summarise(stim = first(currentE), 
              numRspn = n()) %>% 
    mutate(trialNum = 15 * (block-1) + trial) %>% ungroup()
  # stimulus matrix [trial, stimulus number]
  stim_x = t(matrix(trInfo$stim, 30, 60))
  
  # exclude the responses for the first stimulus 
  data_for_model = data %>% filter(stimNum != 1) 
  trInfo_for_model = trInfo %>% filter(stimNum != 1) %>%
    mutate(rspnIdx = cumsum(numRspn))
  
  # response index
  rspn_start = c(1, trInfo_for_model$rspnIdx[1:nrow(trInfo_for_model)-1] + 1)
  rspn_end = trInfo_for_model$rspnIdx
  
  # weight vector matrix 
  w_index = rep(2:30, 2:30)
  w_start = sapply(2:30, function(x) min(which(w_index == x)))
  w_end = sapply(2:30, function(x) max(which(w_index == x)))
  
  ts = vector()
  ts_inv = vector()
  for (i in 2:30) {
    ts = c(ts, 1:i)
    ts_inv = c(ts_inv, i - 1:i + 1)
  }
  
  dataList = list(
    ## in unit of response
    N = nrow(data_for_model),
    sqrt_t = data_for_model$sqrt_time, # time in a trial (unit of 1/30 sec)
    rspn = data_for_model$rspn, # position of slider (-1 ~ + 1)
    task_n = data_for_model$task, # task (1 = perceptual, 2 = value-based)
    # in unit of unique trial and stimulus
    S = nrow(trInfo_for_model),
    stim_x = stim_x, # stimulus information matrix [trial, stim number]
    trialNum = trInfo_for_model$trialNum, # trial number 
    stimNum = trInfo_for_model$stimNum, # stimulus number 
    numRspn = trInfo_for_model$numRspn, # number of responses
    task_s = trInfo_for_model$task, 
    rspn_start = rspn_start, 
    rspn_end = rspn_end,
    # temporal weights
    W = length(w_index),
    w_start = w_start, 
    w_end = w_end,
    ts = ts, 
    ts_inv = ts_inv
  )
  
  
  initf <- function() {list(e_p = runif(1, 0.2, 0.6), 
                            e_r = runif(1, 0.2, 0.6),
                            sigma_w_raw = runif(2, -0.5, 0.5))}
  
  model_out = rstan::sampling(stan_model,
                              data = dataList, 
                              init = initf,
                              warmup = 1000, iter = 5000, chains = 4, cores = 4, 
                              include = FALSE, 
                              pars = c("mu", "sigma", "w"), # parameters to exclude
                              control=list(adapt_delta=0.99,
                                           stepsize = 0.01, max_treedepth = 15))
  
  saveRDS(model_out, paste("ADM2", "fit", sprintf("ADM2_S%02d_fit.RData", subjID),
                           sep = "/"))
}

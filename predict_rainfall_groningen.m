close all
load('raindata_groningen.mat')
Rainfall = table2array(groningendaily);
Rainfall = Rainfall(14:2014);   %to make the code run faster:
Rh = Rainfall;
Rd = Rainfall;
Rw = averages(Rainfall,7);
Rm = averages(Rainfall,30);
AR_order = 4; MA_order = 0;

%split the data into test and train data
data = Rh(1:end-1);
L = length(data);
Lmid = round(L/2);
train_data_h = data(1:Lmid);
test_data_h = data(Lmid+1:end);

data = Rd(1:end-1);
L = length(data);
Lmid = round(L/2);
train_data_d = data(1:Lmid);
test_data_d = data(Lmid+1:end);

data = Rw(1:end-1);
L = length(data);
Lmid = round(L/2);
train_data_w = data(1:Lmid);
test_data_w = data(Lmid+1:end);

data = Rm(1:end-1);
L = length(data);
Lmid = round(L/2);
train_data_m = data(1:Lmid);
test_data_m = data(Lmid+1:end);

%select data type:
data = Rd;
train_data = train_data_d;
test_data = test_data_d;

%%sample PACF:
% subplot(311)
% autocorr(Rd)
% subplot(312)
% autocorr(Rw)
% subplot(313)
% parcorr(Rm)

%make model:
Mdl = arima(AR_order,1,MA_order);
EstMdl = estimate(Mdl,train_data);
[res,~,logL] = infer(EstMdl,data);
stdr = res/sqrt(EstMdl.Variance);

%residual test for this model:
figure('Name','Residual Test')
subplot(2,3,1)
plot(stdr)
title('Standardized Residuals')
subplot(2,3,2)
histogram(stdr,10)
title('Standardized Residuals')
subplot(2,3,3)
autocorr(stdr)
subplot(2,3,4)
parcorr(stdr)
subplot(2,3,5)
qqplot(stdr)

%select order using BIC and AIC information criteria
% max_ar = 5;
% max_ma = 5;
% [AR_order,MA_order] = ARMA_Order_Select(train_data,max_ar,max_ma,1);


%make ARMA model and estimate data:
Fdata = estimate_test(test_data,train_data,AR_order,MA_order);

%forecasting:
step = 20;
[forData,YMSE] = forecast(EstMdl,step,'Y0',test_data);
lower = forData - 1.96*sqrt(YMSE);
upper = forData + 1.96*sqrt(YMSE);

figure()
plot(test_data,'Color',[.7,.7,.7]);
hold on
h1 = plot(length(test_data):length(test_data)+step,[test_data(end);lower],'r:','LineWidth',2);
plot(length(test_data):length(test_data)+step,[test_data(end);upper],'r:','LineWidth',2)
h2 = plot(length(test_data):length(test_data)+step,[test_data(end);forData],'k','LineWidth',2);
legend([h1 h2],'95% Confidence Interval','Predictive Value','Location','NorthWest')
title('Forecast')
hold off

%lead time:
step = 5;
rreal = test_data(step+1:length(Fdata)+step);
rpredict = Fdata;
M = step;

%Mean square error from estimation of test data:
MSE = lt_mse(rreal, rpredict, M);

figure()
plot(rpredict(:,1)); hold on; plot(rreal);
xlabel('Time')
ylabel('Rainfall (mm)')
title('Rainfall prediction on test data (daily)')
legend('measured','predicted')
figure()
plot(2 - (MSE./MSE(1)));
xlabel('Lead time')
ylabel('Efficiency')
title('Efficiency coefficient based on MSE of lead time')


%This function makes the forecast based on the specified ARMA model and the train data, and estimates
%the test data.
function Fdata = estimate_test(test_data,train_data,AR_order,MA_order)
%estimate arma model parameters
Mdl = arima(AR_order,1,MA_order);
EstMdl = estimate(Mdl,train_data);

%forecasting:
step = 5;
N = floor(length(test_data)/step);
Fdata = zeros((N-2)*step,step);

%iterative over all test data for predictions
for i = step:(N-1)*step
    [Fdata(i-step+1,:),~] = forecast(EstMdl,step,'Y0',test_data(1:i));
    round(i/((N-1)*step)*100)
end

end

%This function averages data over N number of adjacent indices
function adj_av = averages(x,N)
x = x(:);   %makes a column vector;
L = length(x);
adj_av = zeros(ceil(L/N),1);
K = length(adj_av);

for n=1:(K-1)
    adj_av(n) = N*(mean(x((1:N) +(n-1)*N)));
end

end

%This function computes the error for lead times 1:M by making a hankel
%matrix of the forecasted data and compute the MSE from comparing to GT.
function error = lt_mse(rreal, rpredict, M)
%{  
calculate MSE per lead time (lt)
    - rreal; the vector of real (measurements) values
    - rpredict; the matrix of p by M representing the predicted values
                in the columns
    - M; the maximum lead time and length of the rowvectors of rpredict
%}

% make hankel matrix from rreal vector with M columns
c = rreal;
r = [rreal(end); rreal(1:M-1)];
h = hankel(c,r);

% calculate MSE per lead time 
error = mean((rpredict - h).^2 , 1);
end


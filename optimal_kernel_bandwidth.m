function [dts, cn_s] = optimal_kernel_bandwidth(spike_times, trial_count)
% Estimation of the optimal kernel width when using a box kernel following
% Shimazaki and Shinomoto 2007a
% Function tests a range of bandwiths and returns the costs associated with
% each. The optimal bandwidth leads to the lowest costs.

dts = (0.00025:0.0001:0.05);
c_nsx = zeros(size(dts));
index = 1;
for dt = dts
    bin_edges = (dt:dt:10);
    n = hist(spike_times, bin_edges);
    k = mean(n);
    v = mean((n-k).^2);
    c_n = (2 * k - v)/((trial_count * dt)^2);
    c_nsx(index) = c_n;
    index = index + 1;
end
cn_s=c_nsx;
end
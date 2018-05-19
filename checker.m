load rat8.mat
v1 = Labels;
load test8.mat
v2 = logical(Labels);
N = length(v1(1,:));
cc = 0;
for i = 1:N
    if v1(:,i) == v2(:,i)
        cc = cc +1;
    end
end
k = N-cc;
disp(1-k/N);
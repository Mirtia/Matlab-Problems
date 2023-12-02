% Myrsini Gkolemi

% Load images from img folder
images = imageDatastore('img\task_05');
images_length = length(images.Files);
X = cell(1, images_length);

% Initialize images X array
for i = 1:images_length
    img = im2double(readimage(images, i)); % normalize image to [0 1]
    X{i} = img;
end

% `a` matrix values
a_vec = [0.45 0.49 0.495];
D = cell(1, 3); % difference
mean_abs_diff_YZ = cell(1, 3); % mean absolute difference
mean_diff_Kpi_magn_c = cell(1, 3);
mean_diff_Kpi_A_c = cell(1, 3);
% ==Phase role==

for x = 1:3
    [xR, ~] = size(X{x}); % image size
    max_X = max(X{x}(:)); % X maximum
    min_X = min(X{x}(:)); % X minimum
    fft2_X = fft2(X{x}); % image fft
    fft2_angle = angle(fft2_X); % angle, phase
    fft2_magnitude = abs(fft2_X); % magnitude
    % fft2_img_polar = fft2_magnitude.*exp(1i * fft2_angle); % polar form
    fft2_shift = fftshift(fft2_X);

    figure('Name', ...
        'Figure 1. Image(' + num2str(x) + ') Magnitude of Fast Fourier Transform of Original image (logscale)');
    imshow(real(log(abs(fft2_shift) + 1)));
    title('1. Image ' + num2str(x) + '. Magnitude of Fast Fourier Transform of Original image (logscale)');
    % Save
    filename = 'magnitudefft' + num2str(x) + '.jpg';
    saveas(gcf, filename);

    figure('Name', ...
        'Figure 1. Image(' + num2str(x) + ') Angle of Fast Fourier Transform of Original image (logscale)');
    imshow(angle(fft2_shift));
    title('1. Image ' + num2str(x) + '. Angle of Fast Fourier Transform of Original image (logscale)');
    % Save
    filename = 'anglefft' + num2str(x) + '.jpg';
    saveas(gcf, filename);

    Y = cell(1, 3); % ifft with A(u,v) magnitude matrix
    Z = cell(1, 3); % linear transformation of Y matrix

    figure('Name', '2. IFFT Images with A(u,v) magnitude and original phase with a = [0.45 0.49 0.495]');
    subplot(4, 2, 1), imshow(X{x});
    title('Original image');
    subplot(4, 2, 2), imhist(X{x});
    mean_abs_diff_YZ_vec = zeros(1, 3);

    for a = 1:3
        A_magnitude = get_A_magnitude(X{x}, a_vec(a)); % create A(u,v) matrix
        fft2_A = A_magnitude .* exp(1i * fft2_angle); % polar form Fourier Transform
        Y{a} = ifft2(fft2_A, 'symmetric'); % ifft
        max_Y = max(Y{a}(:)); % maximum y
        min_Y = min(Y{a}(:)); % minimum y

        % linear transformation
        Z{a} = max_X * ((Y{a} - min_Y) / (max_Y - min_Y)) + min_X;
        % figure, imshow(mat2gray(real(Z{a})));
        hist_X = imhist(X{x});
        % hist_Z = imhist(Z{a});
        equalized_Z = histeq(Z{a}, hist_X);
        subplot(4, 2, 2 * a + 1), imshow(mat2gray(real(equalized_Z)));
        title('a = ' + num2str(a_vec(a)));
        subplot(4, 2, 2 * a + 2), imhist(equalized_Z);

        imhist(equalized_Z);
        diff = abs(X{x} - equalized_Z); % absolute difference
        mean_abs_diff_YZ_vec(1, a) = mean(diff(:)); % mean absolute difference
    end

    mean_abs_diff_YZ{x} = mean_abs_diff_YZ_vec;
    suptitle('2. IFFT Images with A(u,v) magnitude and original phase with a = 0.45 0.49 0.495');

    % Save
    filename = 'auv_' + num2str(x) + '.jpg';
    saveas(gcf, filename);
    % ======================

    % ==Quantization==
    a = 0.495;
    Kphi = [5 9 17 33 65];
    mean_diff_Kpi_A = zeros(1, 5);
    mean_diff_Kpi_magn = zeros(1, 5);

    figure('Name', '2. Quantization');

    for f = 1:5
        % linspace
        phase_q = linspace(min(fft2_angle(:)), max(fft2_angle(:)), Kphi(f));
        quant = imquantize(fft2_angle, phase_q, [0 phase_q]); % phase quantization

        %         figure('Name', '3a. Angle after quantization. phase = ' + ...
        %             num2str(Kphi(f))), imshow(quant, []);
        %         title('3a. Angle after quantization. phase = ' + ...
        %             num2str(Kphi(f)));
        %
        %         % Save
        %         filename = 'quantization_' + num2str(f) + num2str(x) + '.jpg';
        %         saveas(gcf, filename);

        fft2_Kphi_A_img = get_A_magnitude(X{x}, a) .* exp(1i * quant);
        ifft2_Kphi_A_img = ifft2(fft2_Kphi_A_img, 'symmetric');
        diff = abs(ifft2_Kphi_A_img - X{x});
        mean_diff_Kpi_A(1, f) = mean(diff(:));

        subplot(2, 5, f), imshow(mat2gray(ifft2_Kphi_A_img));
        title('A magnitude quantized face f = ' + num2str(Kphi(f)));

        fft2_Kphi_magn_img = fft2_magnitude .* exp(1i * quant);
        ifft2_Kphi_magn_img = ifft2(fft2_Kphi_magn_img, 'symmetric');
        diff = abs(ifft2_Kphi_magn_img - X{x});
        mean_diff_Kpi_magn(1, f) = mean(diff(:));

        subplot(2, 5, f + 5), imshow(mat2gray(ifft2_Kphi_magn_img));
        title('Original magnitude quantized face f = ' + num2str(Kphi(f)));
    end

    % ======================
    suptitle('2. Quantization with A and original magnitude and quantized phase Kphi = [5 9 17 33 65]');
    mean_diff_Kpi_magn_c{x} = mean_diff_Kpi_magn;
    mean_diff_Kpi_A_c{x} = mean_diff_Kpi_A;
    % ==Image compression==
    sorted_fft = sort(abs(fft2_X(:))); % sort
    p_vec = [0.025 0.05 0.075];
    figure('Name', '5. Image Compression p%');
    subplot(2, 2, 1), imshow(X{x});
    title('Original image');

    mean_absolute_diff = cell(1, 3);

    for p = 1:3
        low_bound = sorted_fft(floor((1 - p_vec(p)) * length(sorted_fft)));
        indices = fft2_magnitude > low_bound;
        res_mat = fft2_X .* indices;
        p_img = ifft2(res_mat);
        absolute_difference_p = abs(X{x} - p_img);
        mean_absolute_diff{p} = mean(absolute_difference_p(:));
        subplot(2, 2, 1 + p), imshow(p_img);
        title('p = ' + num2str(p_vec(p)));
    end

    suptitle('5. Image Compression p%');
    % Save
    filename = 'img_compr' + num2str(x) + '.jpg';
    saveas(gcf, filename);

    D{x} = mean_absolute_diff;
    % ======================

end

% ==Functions==

function [A] = get_A_magnitude(img, a)
    % u = m / M , v = n / N
    [N, M] = size(img);
    A = zeros(N, M);

    for m = 1:M
        for n = 1:N
            A(m, n) = 1 / (1 - a * (cos(2 * pi * ((m - 1) / M)) + ...
                cos(2 * pi * ((n - 1) / N))));
        end
    end
    
end

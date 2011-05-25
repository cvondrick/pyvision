function showModel(ww,sc)

if nargin < 2,
  %Set the scale so that the maximum weight is 255
  sc = max(abs(ww(:)));
end

sc = 255/sc;

siz = 20;

im1 = HOGpicture( ww,siz)*sc;
im2 = HOGpicture(-ww,siz)*sc;

%Combine into 1 image
buff = 10;
im1 = padarray(im1,[buff buff],200,'both');
im2 = padarray(im2,[buff buff],200,'both');
im = cat(2,im1,im2);
im = uint8(im);
imagesc(im); colormap gray;

function im = HOGpicture(w, bs)
% HOGpicture(w, bs)
% Make picture of positive HOG weights.

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);    
w(w < 0) = 0;    
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;          
    for k = 1:9,
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
    end
  end
end

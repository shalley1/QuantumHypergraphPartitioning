
imb(P, w) = P * Diagonal(w) * P'
var(P, w) = Diagonal(P * w) - imb(P, w)
varl(P, w) = [w[i] * (Diagonal(P[:, i]) - P[:, i] * P[:, i]') for i in axes(P, 2)]
imbl(P, w) = [w[i] * P[:, i] * P[:, i]' for i in axes(P, 2)]
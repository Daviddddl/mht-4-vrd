from math import log as ln
from math import sqrt, pi

from numpy.linalg import det
from numpy.linalg import inv

from track_utils import gaussian_bbox


class KFilter:
    """Kalman-filter target."""

    def __init__(self, model, x0, P0):
        """Init."""
        self.model = model
        self.x = x0
        self.P = P0
        self.trace = [(x0, P0)]
        self._calc_bbox()

    def __repr__(self):
        """Return string representation of measurement."""
        return "T({}, P)".format(self.x)

    def predict(self, dT):
        """Perform motion prediction."""
        new_x, new_P = self.model(self.x, self.P, dT)
        self.trace.append((new_x, new_P))
        self.x, self.P = new_x, new_P

        self._calc_bbox()

    def correct(self, r):
        """Perform correction (measurement) update."""
        zhat, H = r.mfn(self.x)
        dz = r.z - zhat
        S = H @ self.P @ H.T + r.R
        SI = inv(S)
        K = self.P @ H.T @ SI
        self.x += K @ dz
        self.P -= K @ H @ self.P

        score = dz.T @ SI @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))

        self._calc_bbox()

        return float(score)

    def nll(self, r):
        """Get the nll score of assigning a measurement to the filter."""
        zhat, H = r.mfn(self.x)
        dz = r.z - zhat
        S = H @ self.P @ H.T + r.R
        score = dz.T @ inv(S) @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)

    def _calc_bbox(self, nstd=2):
        """Calculate minimal bounding box approximation."""
        self._bbox = gaussian_bbox(self.x[0:2], self.P[0:2, 0:2])

    def bbox(self):
        """Get minimal bounding box approximation."""
        return self._bbox

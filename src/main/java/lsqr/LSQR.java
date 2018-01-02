package lsqr;

import com.github.fommil.netlib.BLAS;
import java.util.Arrays;

public class LSQR {

    private static double eps = 1e-16;

    private static BLAS blas = BLAS.getInstance();

    /**
     * Find the least-squares solution to a large, sparse, linear system
     of equations.
     The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
     ``min ||Ax - b||^2 + d^2 ||x||^2``.
     The matrix A may be square or rectangular (over-determined or
     under-determined), and may have any rank.
     ::
     1. Unsymmetric equations --    solve  A*x = b
     2. Linear least squares  --    solve  A*x = b
     in the least-squares sense
     3. Damped least squares  --    solve  (   A    )*x = ( b )
     ( damp*I )     ( 0 )
     in the least-squares sense
     Parameters
     ----------
     A : {sparse matrix, ndarray, LinearOperatorLinear}
     Representation of an m-by-n matrix.  It is required that
     the linear operator can produce ``Ax`` and ``A^T x``.
     b : (m,) ndarray
     Right-hand side vector ``b``.
     damp : float
     Damping coefficient.
     atol, btol : float
     Stopping tolerances. If both are 1.0e-9 (say), the final
     residual norm should be accurate to about 9 digits.  (The
     final x will usually have fewer correct digits, depending on
     cond(A) and the size of damp.)
     conlim : float
     Another stopping tolerance.  lsqr terminates if an estimate of
     ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
     b``, `conlim` could be as large as 1.0e+12 (say).  For
     least-squares problems, conlim should be less than 1.0e+8.
     Maximum precision can be obtained by setting ``atol = btol =
     conlim = zero``, but the number of iterations may then be
     excessive.
     iter_lim : int
     Explicit limitation on number of iterations (for safety).
     show : bool
     Display an iteration log.
     calc_var : bool
     Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
     Returns
     -------
     x : ndarray of float
     The final solution.
     istop : int
     Gives the reason for termination.
     1 means x is an approximate solution to Ax = b.
     2 means x approximately solves the least-squares problem.
     itn : int
     Iteration number upon termination.
     r1norm : float
     ``norm(r)``, where ``r = b - Ax``.
     r2norm : float
     ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
     ``damp == 0``.
     anorm : float
     Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
     acond : float
     Estimate of ``cond(Abar)``.
     arnorm : float
     Estimate of ``norm(A'*r - damp^2*x)``.
     xnorm : float
     ``norm(x)``
     var : ndarray of float
     If ``calc_var`` is True, estimates all diagonals of
     ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
     damp^2*I)^{-1}``.  This is well defined if A has full column
     rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
     < n`` and ``damp = 0.``)
     Notes
     -----
     LSQR uses an iterative method to approximate the solution.  The
     number of iterations required to reach a certain accuracy depends
     strongly on the scaling of the problem.  Poor scaling of the rows
     or columns of A should therefore be avoided where possible.
     For example, in problem 1 the solution is unaltered by
     row-scaling.  If a row of A is very small or large compared to
     the other rows of A, the corresponding row of ( A  b ) should be
     scaled up or down.
     In problems 1 and 2, the solution x is easily recovered
     following column-scaling.  Unless better information is known,
     the nonzero columns of A should be scaled so that they all have
     the same Euclidean norm (e.g., 1.0).
     In problem 3, there is no freedom to re-scale if damp is
     nonzero.  However, the value of damp should be assigned only
     after attention has been paid to the scaling of A.
     The parameter damp is intended to help regularize
     ill-conditioned systems, by preventing the true solution from
     being very large.  Another aid to regularization is provided by
     the parameter acond, which may be used to terminate iterations
     before the computed solution becomes very large.
     If some initial estimate ``x0`` is known and if ``damp == 0``,
     one could proceed as follows:
     1. Compute a residual vector ``r0 = b - A*x0``.
     2. Use LSQR to solve the system  ``A*dx = r0``.
     3. Add the correction dx to obtain a final solution ``x = x0 + dx``.
     This requires that ``x0`` be available before and after the call
     to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
     to solve A*x = b and k2 iterations to solve A*dx = r0.
     If x0 is "good", norm(r0) will be smaller than norm(b).
     If the same stopping tolerances atol and btol are used for each
     system, k1 and k2 will be similar, but the final solution x0 + dx
     should be more accurate.  The only way to reduce the total work
     is to use a larger stopping tolerance for the second system.
     If some value btol is suitable for A*x = b, the larger value
     btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
     Preconditioning is another way to reduce the number of iterations.
     If it is possible to solve a related system ``M*x = b``
     efficiently, where M approximates A in some helpful way (e.g. M -
     A has low rank or its elements are small relative to those of A),
     LSQR may converge more rapidly on the system ``A*M(inverse)*z =
     b``, after which x can be recovered by solving M*x = z.
     If A is symmetric, LSQR should not be used!
     Alternatives are the symmetric conjugate-gradient method (cg)
     and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
     applies to any symmetric A and will converge more rapidly than
     LSQR.  If A is positive definite, there are other implementations
     of symmetric cg that require slightly less work per iteration than
     SYMMLQ (but will take the same number of iterations).
     References
     ----------
     .. [1] C. C. Paige and M. A. Saunders (1982a).
     "LSQR: An algorithm for sparse linear equations and
     sparse least squares", ACM TOMS 8(1), 43-71.
     .. [2] C. C. Paige and M. A. Saunders (1982b).
     "Algorithm 583.  LSQR: Sparse linear equations and least
     squares problems", ACM TOMS 8(2), 195-209.
     .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
     systems using LSQR and CRAIG", BIT 35, 588-604.

     * @param a
     * @param m
     * @param b
     * @param damp
     * @param atol
     * @param btol
     * @param conlim
     * @param iter_lim
     * @param calc_var
     * @param x0
     * @return
     */
    public static double[] lsqr(double[] a, int m, double[] b, double damp, double atol, double btol, double conlim, double iter_lim, boolean calc_var, double[] x0) {
        int n = a.length / m;
        if (iter_lim < 0)
            iter_lim = 2 * n;
        double[] var = new double[n];
        double itn = 0;
        double istop = 0;
        double ctol = 0;
        if (conlim > 0)
            ctol = 1 / conlim;
        double anorm = 0;
        double acond = 0;
        double dampsq = Math.pow(damp, 2.0);
        double ddnorm = 0;
        double res2 = 0;
        double xnorm = 0;
        double xxnorm = 0;
        double z = 0;
        double cs2 = -1;
        double sn2 = 0;

        // Set up the first vectors u and v for the bidiagonalization.
        // These satisfy  beta*u = b - A*x,  alfa*v = A'*u.
        double[] u = Arrays.copyOf(b, b.length);
        double bnorm = blas.dnrm2(b.length, b, 1);
        double[] x;
        double beta;
        if (x0 == null) {
            x = new double[n];
            beta = bnorm;
        }
        else {
            x = x0;
            blas.dgemv("N", m, a.length / m, -1.0, a, m, x, 1, 1.0, u, 1);
            beta = blas.dnrm2(u.length, u, 1);
        }
        double[] v = new double[a.length / m];
        double alfa;
        if (beta > 0) {
            blas.dscal(u.length, 1 / beta, u, 1);
            blas.dgemv("T", m, n, 1.0, a, m, u, 1, 0, v, 1);
            alfa = blas.dnrm2(v.length, v, 1);
        }
        else {
            v = Arrays.copyOf(x, x.length);
            alfa = 0;
        }

        if (alfa > 0)
            blas.dscal(v.length, 1 / alfa, v, 1);
        double[] w = Arrays.copyOf(v, v.length);

        double rhobar = alfa;
        double phibar = beta;
        double rnorm = beta;
        double r1norm = rnorm;
        double r2norm = rnorm;
        double arnorm = alfa * beta;
        double[] dk = new double[w.length];
        if (arnorm == 0)
            return x;//, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

        // Main iteration loop.
        while (itn < iter_lim) {
            itn = itn + 1;
            // Perform the next step of the bidiagonalization to obtain the
            // next  beta, u, alfa, v.  These satisfy the relations
            //            beta*u  =  A*v   -  alfa*u,
            //            alfa*v  =  A'*u  -  beta*v.
            blas.dgemv("N", m, n, 1.0, a, m, v, 1, -alfa, u, 1);
            beta = blas.dnrm2(u.length, u, 1);

            if (beta > 0) {
                blas.dscal(u.length, 1 / beta, u, 1);
                anorm = Math.sqrt(Math.pow(anorm, 2) + Math.pow(alfa, 2) + Math.pow(beta, 2) + Math.pow(damp, 2));
                blas.dgemv("T", m, a.length / m, 1.0, a, m, u, 1, -beta, v, 1);
                alfa = blas.dnrm2(v.length, v, 1);
                if (alfa > 0)
                    blas.dscal(v.length, 1 / alfa, v, 1);
            }

            // Use a plane rotation to eliminate the damping parameter.
            // This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
            double rhobar1 = Math.sqrt(Math.pow(rhobar, 2) + Math.pow(damp, 2));
            double cs1 = rhobar / rhobar1;
            double sn1 = damp / rhobar1;
            double psi = sn1 * phibar;
            phibar = cs1 * phibar;

            // Use a plane rotation to eliminate the subdiagonal element (beta)
            // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
            double[] symOrtho = symOrtho(rhobar1, beta);
            double cs = symOrtho[0];
            double sn = symOrtho[1];
            double rho = symOrtho[2];

            double theta = sn * alfa;
            rhobar = -cs * alfa;
            double phi = cs * phibar;
            phibar = sn * phibar;
            double tau = sn * phi;

            double t1 = phi / rho;
            double t2 = -theta / rho;
            blas.dcopy(w.length, w, 1, dk, 1);
            blas.dscal(dk.length, 1 / rho, dk, 1);

            blas.daxpy(w.length, t1, w, 1, x, 1);
            blas.daxpy(w.length, t2, w, 1, v, 1);
            ddnorm = ddnorm + Math.pow(blas.dnrm2(dk.length, dk, 1), 2);

            if (calc_var)
                blas.daxpy(var.length, 1.0, pow(dk, 2), 1, var, 1);

            // Use a plane rotation on the right to eliminate the
            // super-diagonal element (theta) of the upper-bidiagonal matrix.
            // Then use the result to estimate norm(x).
            double delta = sn2 * rho;
            double gambar = -cs2 * rho;
            double rhs = phi - delta * z;
            double zbar = rhs / gambar;
            xnorm = Math.sqrt(xxnorm + Math.pow(zbar, 2));
            double gamma = Math.sqrt(Math.pow(gambar, 2) + Math.pow(theta, 2));
            cs2 = gambar / gamma;
            sn2 = theta / gamma;
            z = rhs / gamma;
            xxnorm = xxnorm + Math.pow(z, 2);

            // Test for convergence.
            // First, estimate the condition of the matrix  Abar,
            // and the norms of  rbar  and  Abar'rbar.
            acond = anorm * Math.sqrt(ddnorm);
            double res1 = Math.pow(phibar, 2);
            res2 = res2 + Math.pow(psi, 2);
            rnorm = Math.sqrt(res1 + res2);
            arnorm = alfa * Math.abs(tau);

            // Distinguish between
            //    r1norm = ||b - Ax|| and
            //    r2norm = rnorm in current code
            //           = sqrt(r1norm^2 + damp^2*||x||^2).
            //    Estimate r1norm from
            //    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
            // Although there is cancellation, it might be accurate enough.
            double r1sq = Math.pow(rnorm, 2) - dampsq * xxnorm;
            r1norm = Math.sqrt(Math.abs(r1sq));
            if (r1sq < 0)
                r1norm = -r1norm;
            r2norm = rnorm;

            // Now use these norms to estimate certain other quantities,
            // some of which will be small near a solution.
            double test1 = rnorm / bnorm;
            double test2 = arnorm / (anorm * rnorm + eps);
            double test3 = 1 / (acond + eps);
            t1 = test1 / (1 + anorm * xnorm / bnorm);
            double rtol = btol + atol * anorm * xnorm / bnorm;

            // The following tests guard against extremely small values of
            // atol, btol  or  ctol.  (The user may have set any or all of
            // the parameters  atol, btol, conlim  to 0.)
            // The effect is equivalent to the normal tests using
            // atol = eps,  btol = eps,  conlim = 1/eps.
            if (itn >= iter_lim)
                istop = 7;
            if (1 + test3 <= 1)
                istop = 6;
            if (1 + test2 <= 1)
                istop = 5;
            if (1 + t1 <= 1)
                istop = 4;

            // Allow for tolerances set by the user.
            if (test3 <= ctol)
                istop = 3;
            if (test2 <= atol)
                istop = 2;
            if (test1 <= rtol)
                istop = 1;

            if (istop != 0)
                break;
        }
        return x;//, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var }
    }

    private static double[] symOrtho(double a, double b) {
        if (b == 0)
            return new double[]{ Math.signum(a), 0, Math.abs(a) };
        else if (a == 0)
            return new double[]{ 0, Math.signum(b), Math.abs(b) };
        else {
            double c, s, r;
            if (Math.abs(b) > Math.abs(a)) {
                double tau = a / b;
                s = Math.signum(b) / Math.sqrt(1 + tau * tau);
                c = s * tau;
                r = b / s;
            }
            else {
                double tau = b / a;
                c = Math.signum(a) / Math.sqrt(1 + tau * tau);
                s = c * tau;
                r = a / c;
            }
            return new double[]{ c, s, r };
        }
    }

    private static double[] pow(double[] a, double pow) {
        double[] res = new double[a.length];
        for (int i = 0; i < res.length; i++)
            res[i] = Math.pow(a[i], pow);
        return res;
    }
}

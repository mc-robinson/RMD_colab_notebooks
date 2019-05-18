import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RMD(object):
    """Random Matrix Discriminant (RMD)

    Uses random matrix theory to separate classes for *binary* classification.
    RMD is particulalry useful in undersampled problems (N < p) because
    spurious correlations between the large number of features will be detected
    and filtered out before prediction.

    Essentially, RMD finds the principal components (PCs) for each class of
    samples representing signal, and not noise. Each class is then represented
    by a subspace of significant PCs.

    During prediction, the distance of a sample to the subspace for each class
    is calculated. The closer subspace represents the more likely class for the
    sample, as is reflected in the provided score.


    Parameters
    ------------
    None.

    Attributes
    -----------
    pos_label_ : label used to indicate positive class
    neg_label_ : label used to indicate negative class
    pos_X_train_ : numpy array of training samples from positive class
    neg_X_train_ : numpy array of training samples from negative class
    fit_ : Boolean indicating if the classifier has been fit

    """

    def __init__(self):
        self.fit_ = False

    def fit(self, X, y, pos_label=None):
        """Fit training data.

        The word 'fit' here is perhaps misleading. Our implementation makes it
        easiet to construct RMD as a 'lazy learner.' Training data is only used
        at prediction time, and this function simply stores the data for later.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values. Must be binary.
        pos_label : int or str
          The label indicating the positive class.
          Only necessary if positive label is not 1.

        Returns
        -------
        self : object

        """

        # check input
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                'Only binary target data is currently supported.'
                f'{len(classes)} have been detected.')
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError(
                'Data is not binary and pos_label is not specified')
        elif pos_label is None:
            self.pos_label_ = 1.
        else:
            self.pos_label_ = pos_label

        self.neg_label_ = classes[classes!=self.pos_label_][0]

        # save matrices for use during prediction
        self.X_train_ = X
        self.pos_X_train_ = X[y==self.pos_label_]
        self.neg_X_train_ = X[y!=self.pos_label_]

        self.fit_ = True

    def predict_scores(self, X, scoring='basic'):
        """Return score indicating likely class of sample.

        The returned scores are computed as:
            (distance to subspace of negative class /
            (total distance to subspaces of both classes))

        Therefore, a higher score indicates that the sample is further away
        from the negative class, and thus likely belongs to the positive class.

        Ideally, any score > 0.5 would correspond to negative class membership.
        However, this is often not the case. Classes are often well separated
        by scores, despite 0.5 not being an appropriate classification
        threshold. Therefore, it is suggest that a user pick a score threshold
        that adequetly meets their desired balance of false/true positives.

        The AUC performance of the classification scores can be easily tested:
        ```
        from sklearn import metrics
        test_scores = rmd.predict_scores(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, test_scores)
        roc_auc = auc(fpr, tpr)
        ```

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Test vectors, where n_samples is the number of samples and
          n_features is the number of features.

        Returns
        -------
        scores : numpy array
          Scores indicating likelihood of membership to the negative class.
          (should not be interpreted as a strict probability)

        """

        if not self.fit_:
            raise ValueError('The classifier has not been fit yet.')

        dists_to_pos_subspace, pos_evals, pos_evecs = \
        self._get_distance_to_train_set(self.pos_X_train_, X, get_eigs=True)

        dists_to_neg_subspace, neg_evals, neg_evecs = \
        self._get_distance_to_train_set(self.neg_X_train_, X, get_eigs=True)

        scores = []
        for i in range(X.shape[0]):
            dist_to_pos_subspace = dists_to_pos_subspace[i]
            dist_to_neg_subspace = dists_to_neg_subspace[i]
            if scoring == 'basic':
                total_dist = dist_to_pos_subspace + dist_to_neg_subspace
                # get rid of imaginary part
                score = np.real(dist_to_neg_subspace/total_dist)
            elif scoring == 'weighted':
                n_pos = self.pos_X_train_.shape[0]
                n_neg = self.neg_X_train_.shape[0]
                n_total = n_pos + n_neg
                score = np.real(((n_neg/n_total)*dist_to_neg_subspace) -
                              ((n_pos/n_total)*dist_to_pos_subspace))
            elif scoring == 'dist_to_positive':
                score = -1*dist_to_pos_subspace
            elif scoring == 'dist_to_negative':
                score = dist_to_neg_subspace
            elif scoring == 'theoretical':
                n_pos = self.pos_X_train_.shape[0]
                n_neg = self.neg_X_train_.shape[0]
                p_pos = len(pos_evecs[0])
                p_neg = len(neg_evecs[0])
                pos_lambda_max = np.max(pos_evals)
                neg_lambda_max = np.max(neg_evals)

                pos_scale = 1/np.sqrt(1+p_pos/(n_pos*pos_lambda_max))
                neg_scale = 1/np.sqrt(1+p_neg/(n_neg*neg_lambda_max))

                score = np.real((neg_scale*dist_to_neg_subspace) -
                              (pos_scale*dist_to_pos_subspace))
            else:
                raise ValueError('Not a valid scoring method')
            # score = dist_to_neg_subspace - dist_to_pos_subspace
            scores.append(score)

        return np.array(scores)

    def get_train_features(self):

        if not self.fit_:
            raise ValueError('The classifier has not been fit yet.')

        pos_train_features = \
            self._get_projections(self.pos_X_train_, self.X_train_)
        neg_train_features = \
            self._get_projections(self.neg_X_train_, self.X_train_)

        train_features = np.concatenate(
                            (pos_train_features, neg_train_features), axis=1)

        return np.real(train_features)

    def get_test_features(self, X_test):

        if not self.fit_:
            raise ValueError('The classifier has not been fit yet.')

        pos_test_features = self._get_projections(self.pos_X_train_, X_test)
        neg_test_features = self._get_projections(self.neg_X_train_, X_test)

        test_features = np.concatenate(
                            (pos_test_features, neg_test_features), axis=1)

        return np.real(test_features)

    def _get_projections(self, X_train, X_test):

        ### CAN PROBABLY ADAPT OTHER FUNCTIONS TO INCLUDE THIS ###
        train_set, test_set = \
            self._delete_uninformative_columns(X_train, X_test)

#         if X_test is not None:
#             ### NEED TO FIGURE OUT WHICH IS BETTER ###
#             ### NORMALIZE WRT TRAIN OR TEST ###
#             ### RIGHT NOW DOING IT BEFORE DELETE ZERO ROWS ###
#             test_set = ((test_set - np.mean(train_set, axis=0)) /
#                            np.std(train_set, axis=0))

        train_set = self._add_zero_rows(train_set)

        # standardize all matrices with respect to train_set mean and std
        train_set = ((train_set - np.mean(train_set, axis=0)) /
                     np.std(train_set, axis=0))
        test_set = ((test_set - np.mean(train_set, axis=0)) /
                    np.std(train_set, axis=0))

        # get important evals and evecs of train_set
        evals, evecs = self._get_significant_eig(train_set)

        # would be nice to find a way to normalize dists
        # so that distances to positive and negative classes are comparable
        projections = (test_set@evecs)#@(evecs.T)

        return projections


    def plot_evals(self):
        """Plot histograms showing distribution of eigenvalues for each class.

        In each histogram, the MP-bound is also shown. Eigenvectors
        corresponding to eigenvalues greater than the MP-bound are used to
        construct the subspace defining each class. These eigenvectors are
        considered the 'signal' while all others are considered the result of
        noise.

        """
        for label in ['positive', 'negative']:

            if label == 'positive':
                pos_evals, pos_evecs, pos_MP_bound = \
                    self._get_all_eig(self.pos_X_train_)
                pos_evals = np.real(pos_evals)
            elif label == 'negative':
                neg_evals, neg_evecs, neg_MP_bound = \
                    self._get_all_eig(self.neg_X_train_)
                neg_evals = np.real(neg_evals)

            plt.figure(figsize=(6,10))
            if label =='positive':
                plt.subplot(211)
                plt.hist(pos_evals, density=True, bins=100)
                plt.axvline(x=pos_MP_bound,c='red',linestyle='dashed',zorder=10,
                    label='MP bound')
                plt.title('Eigenvalues for Positive Class')
                plt.xlabel('lambda')
                plt.ylabel('density')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlim(-2,50)
                # inset axes....
                axins = plt.gca().inset_axes([0.55,0.55, 0.4, 0.4])
                axins.hist(pos_evals, density=True, bins=100)
                axins.axvline(x=pos_MP_bound,c='red',linestyle='dashed',
                              zorder=10)
                # sub region of the original image
                x1, x2, y1, y2 = (pos_MP_bound-2), 50, 0, 0.02
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels('')
                axins.set_yticklabels('')
                plt.gca().indicate_inset_zoom(axins);

            else:
                plt.subplot(212)
                plt.hist(neg_evals, density=True, bins=100)
                plt.axvline(x=neg_MP_bound,c='red',linestyle='dashed',zorder=10,
                    label='MP bound')
                plt.title('Eigenvalues for Negative Class')
                plt.xlabel('lambda')
                plt.ylabel('density')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlim(-2,50)
                # inset axes....
                axins = plt.gca().inset_axes([0.55,0.55, 0.4, 0.4])
                axins.hist(neg_evals, density=True, bins=100)
                axins.axvline(x=neg_MP_bound,c='red',linestyle='dashed',
                              zorder=10)
                # sub region of the original image
                x1, x2, y1, y2 = (neg_MP_bound-2), 50, 0, 0.02
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels('')
                axins.set_yticklabels('')
                plt.gca().indicate_inset_zoom(axins);

    def _delete_uninformative_columns(self, X_train, X_test):
        """Delete descriptors that the are same for every sample in training set

        These are also the columns for which the variance would be zero.
        This, therefore, avoids divide-by-zero errors when computing the
        correlation matrix. The one catch is that the test set must have the
        exact same columns deleted as the training set.
        """
        # I am using broadcasting to be a bit fancy here
        # note that we also want to keep the columns of all 1s
        X_train_new = X_train[:, ~((X_train == X_train[0,:]).all(axis=0) & (X_train==0).all(axis=0))]
        X_test_new = X_test[:, ~((X_train == X_train[0,:]).all(axis=0) & (X_train==0).all(axis=0))]
        return X_train_new, X_test_new

    def _add_zero_rows(self, X_train):
        max_number_of_ones_in_col = np.max(np.sum(X_train, axis=0))
        if (max_number_of_ones_in_col/(X_train.shape[0]) > 0.5):
            rows_to_add = np.zeros(
                             shape=(max_number_of_ones_in_col, X_train.shape[1])
                          )
            X_train = np.concatenate((X_train, rows_to_add))

        return X_train

    def _get_significant_eig(self, X_train):
        """Use the Marcenko–Pastur (MP) distribution to find the signal.

        The MP distribution describes the eigenvalue distribution of a
        correlation matrix A, where the entries of A ~ N(0,1):

        $$
        \rho(\lambda)=
        \frac{
            \sqrt{
                \left[(1+\sqrt{\gamma})^{2}-\lambda\right]_{+}
                \left[\lambda-(1-\sqrt{\gamma})^{2}\right]_{+}
            }
        }
        {2 \pi \gamma \lambda}
        $$

        where $\lambda$ is an eigenvalue, and $\gamma=p/N$ indicates how well
        sampled the dataset is. (N is the number of samples, p is the number of
        features)

        Therefore, any eigenvalues $\lambda$ above $(1+\sqrt{\gamma})^{2}$
        (where $\rho(\lambda)$ is expected to be 0), correspond to statistically
        significant signals.
        """
        corr_mat = np.corrcoef(X_train.T)
        # replace all nans and infs with 0
        corr_mat[np.isnan(corr_mat)] = 0
        corr_mat[np.isinf(corr_mat)] = 0
        evals, evecs = np.linalg.eig(corr_mat)

        N, p = X_train.shape
        MP_bound = (1 + np.sqrt(p/N))**2

        sig_evals = evals[evals > MP_bound]
        sig_evecs = evecs[:,evals > MP_bound]

        return sig_evals, sig_evecs

    def _get_all_eig(self, train_set):

        train_set, test_set = \
            self._delete_uninformative_columns(train_set, train_set)

        train_set = self._add_zero_rows(train_set)

        # standardize all matrices with respect to train_set mean and std
        train_set = ((train_set - np.mean(train_set, axis=0)) /
                     np.std(train_set, axis=0))
        corr_mat = np.corrcoef(train_set.T)
        evals, evecs = np.linalg.eig(corr_mat)
        N, p = train_set.shape
        MP_bound = (1 + np.sqrt(p/N))**2

        return evals, evecs, MP_bound

    def _get_distance_to_pc_space(self, train_set, test_set, get_eigs=False):
        """Get the distance of test samples to the principle componenent (PC)
        subspace of the training set (usually just one class).

        More formally, if there are $m$ eigenvalues greater than the
        MP-bound, then the linear space spanned by the $m$ associated
        eigenvectors, $\mathbf{V}=\operatorname{span}\left(\mathbf{v}_{1},
        \mathbf{v}_{2}, \cdots \mathbf{v}_{m}\right)$ is the PC subspace of
        interest.

        If we then have a ligand vector $\mathbf{u}$ of unknown class from the
        test set, the projection of this vector $\mathbf{u}$ onto the suspace
        $\mathbf{V}$ is given by:

        $$
        \mathbf{u}_{p} =
        \sum_{i=1}^{m}\left(\mathbf{v}_{i} \cdot \mathbf{u}\right)\mathbf{v}_{i}
        $$

        where $\mathbf{u}_{p}$ is the projection of $\mathbf{u}$ onto
        $\mathbf{V}$. The distance to the subspace is then simply the Euclidian
        norm of the difference, $\left\|\mathbf{u}-\mathbf{u}_{p}\right\|_2$
        """

        # standardize all matrices with respect to train_set mean and std
        train_set = ((train_set - np.mean(train_set, axis=0)) /
                     np.std(train_set, axis=0))
        test_set = ((test_set - np.mean(train_set, axis=0)) /
                        np.std(train_set, axis=0))

        # get important evals and evecs of train_set
        evals, evecs = self._get_significant_eig(train_set)

        # would be nice to find a way to normalize dists
        # so that distances to positive and negative classes are comparable
        projection_matrix = (test_set@evecs)@(evecs.T)
        dist_to_pc_space = \
            np.sqrt(np.sum((projection_matrix-test_set)**2, axis=1))

        if get_eigs:
            return dist_to_pc_space, evals, evecs
        else:
            return dist_to_pc_space

    def _get_distance_to_train_set(self, train_set, test_set, get_eigs=False):
        """Get the distance from compounds in test_set of unknown class to the
        principal component subspace of a train_set of known class

        Before computing the distance, all uninformative features of the
        train_set are deleted.
        """
        train_set, test_set = \
            self._delete_uninformative_columns(train_set, test_set)

        train_set = self._add_zero_rows(train_set)

        dist_to_train_set = \
            self._get_distance_to_pc_space(train_set, test_set, get_eigs)

        return dist_to_train_set

    def _get_effective_corr_mat(self):

        pos_train_set = self._add_zero_rows(self.pos_X_train_)
        neg_train_set = self._add_zero_rows(self.neg_X_train_)

        # standardize all matrices with respect to train_set mean and std
        # add small number in denom to avoid divide by zero
        pos_train_set = ((pos_train_set - np.mean(pos_train_set,axis=0))
                         / (np.std(pos_train_set, axis=0) + 1e-10))

        neg_train_set = ((neg_train_set - np.mean(neg_train_set,axis=0))
                         / (np.std(neg_train_set, axis=0) + 1e-10))


        # note that we do not delete the 'uniformative columns' before getting
        # the evecs, in order to make sure they are the same size.
        pos_evals, pos_evecs = self._get_significant_eig(pos_train_set)
        neg_evals, neg_evecs = self._get_significant_eig(neg_train_set)

        effective_corr_mat = (
            sum([np.outer(x,x) for x in pos_evecs.T])/(pos_evecs.T.shape[0]) -
            sum([np.outer(x,x) for x in neg_evecs.T])/(neg_evecs.T.shape[0])
        )

        return np.real(effective_corr_mat)

    def interpret(self, num_features=10, plot=True):

        if not self.fit_:
            raise ValueError('The classifier has not been fit yet.')

        effective_corr_mat = self._get_effective_corr_mat()
        # top_features = np.argsort(self.X.sum(axis=0))[-num_features:]
        top_features = np.argsort(
                           np.sum(np.abs(effective_corr_mat),axis=0),
                           axis=None
                       )[-num_features:]
        # get indices of biggest correlations
        # then use these to select rows and cols
        small_effective_corr_mat = \
        effective_corr_mat[top_features][:,top_features]

        small_effective_corr_mat = \
        small_effective_corr_mat - np.diag(small_effective_corr_mat)

        if plot:
            import seaborn as sns
            f, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(small_effective_corr_mat,
                        mask=np.zeros_like(small_effective_corr_mat, dtype=np.bool),
                        cmap=sns.diverging_palette(275,150,s=80,l=55,as_cmap=True),
                        xticklabels=top_features,
                        yticklabels=top_features,
                        square=True,
                        cbar=True,
                        center=0.0,
                        ax=ax)
            plt.xlabel("Feature Number")
            plt.ylabel("Feature Number")
        else:
            return small_effective_corr_mat

    def find_feature_fragments(self, feature_num, mols, radius=3, nBits=1024):

        from rdkit import Chem
        from rdkit import DataStructs
        from rdkit.Chem.Fingerprints import FingerprintMols
        from rdkit.Chem import AllChem, DataStructs, Draw
        from collections import defaultdict

        fragmol = defaultdict(list)
        fragmol_mol = defaultdict(list)
        for mol in mols:
            bit_info = {}
            #fragmol = defaultdict( list )
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                       nBits=nBits,
                                                       bitInfo=bit_info)
            for bit, info in bit_info.items():
                for atm_idx, rad in info:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atm_idx)
                    amap = {}
                    try:
                        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    except:
                        raise ValueError('feature does not turn on any bits')
                    smi = Chem.MolToSmiles(submol)
                    if smi != '':
                        if smi not in fragmol[bit]:
                            fragmol[bit].append(smi)
                            fragmol_mol[bit].append(submol)

        return fragmol[feature_num], fragmol_mol[feature_num]

    def cPCA(self, X_test=None):

        from sklearn.decomposition import PCA

        # delete only the columns that are uniformative to both sets
        full_train_set = np.concatenate((self.pos_X_train_, self.neg_X_train_))
        if X_test is not None:
            full_train_set, test_set = \
                self._delete_uninformative_columns(full_train_set,
                                                   X_test)
        else:
            full_train_set, _ = \
                self._delete_uninformative_columns(full_train_set,
                                                   full_train_set)

        pos_train_set = full_train_set[:self.pos_X_train_.shape[0]]
        neg_train_set = full_train_set[self.pos_X_train_.shape[0]:]

        pos_train_set = self._add_zero_rows(pos_train_set)
        neg_train_set = self._add_zero_rows(neg_train_set)

        pos_train_set = ((pos_train_set - np.mean(pos_train_set, axis=0))
                         / (np.std(pos_train_set, axis=0) + 1e-10))

        neg_train_set = ((neg_train_set - np.mean(neg_train_set, axis=0))
                         / (np.std(neg_train_set, axis=0) + 1e-10))

        if X_test is not None:
            test_set = ((test_set - np.mean(test_set, axis=0))
                         / (np.std(test_set, axis=0) + 1e-10))

        pos_corr_mat = np.corrcoef(pos_train_set.T)
        neg_corr_mat = np.corrcoef(neg_train_set.T)
        # replace all nans and infs with 0
        pos_corr_mat[np.isnan(pos_corr_mat)] = 0
        pos_corr_mat[np.isinf(pos_corr_mat)] = 0
        neg_corr_mat[np.isnan(neg_corr_mat)] = 0
        neg_corr_mat[np.isinf(neg_corr_mat)] = 0

        contrast_corr_mat = pos_corr_mat - neg_corr_mat
        c_evals, c_evecs = np.linalg.eig(contrast_corr_mat)

        c_transformed_pos_X = pos_train_set@c_evecs
        c_transformed_neg_X = neg_train_set@c_evecs
        if X_test is not None:
            c_transformed_test_set = test_set@c_evecs

        # now classical PCA
        # note that we do not add zero rows here
        full_train_set = ((full_train_set - np.mean(full_train_set, axis=0))
                         / (np.std(full_train_set, axis=0) + 1e-10))
        pca = PCA(n_components=2)
        pca.fit(full_train_set)
        transformed_full_X = pca.transform(full_train_set)
        if X_test is not None:
            transformed_test = pca.transform(test_set)

        plt.figure(figsize=(6,10))

        #plot cPCA
        plt.subplot(211)
        plt.scatter(c_transformed_pos_X[:, 0],
                    c_transformed_pos_X[:, 1],
                    alpha=0.2, label='pos')
        plt.scatter(c_transformed_neg_X[:, 0],
                    c_transformed_neg_X[:, 1],
                    alpha=0.2, label='neg')
        if X_test is not None:
            plt.scatter(c_transformed_test_set[:, 0],
                    c_transformed_test_set[:, 1],
                    alpha=0.5, color='black', label='test sample')
        plt.xlabel('cPC1')
        plt.ylabel('cPC2')
        plt.title('PCs calculated using contrastive PCA:')
        plt.legend()

        # plot regular PCA
        plt.subplot(212)
        plt.scatter(transformed_full_X[:self.pos_X_train_.shape[0], 0],
                    transformed_full_X[:self.pos_X_train_.shape[0], 1],
                    alpha=0.2, label='pos')
        plt.scatter(transformed_full_X[self.neg_X_train_.shape[0]:, 0],
                    transformed_full_X[self.neg_X_train_.shape[0]:, 1],
                    alpha=0.2, label='neg')
        if X_test is not None:
            plt.scatter(transformed_test[:, 0],
                    transformed_test[:, 1],
                    alpha=0.5, color='black', label='test')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCs calculated using classical PCA:')
        plt.legend()

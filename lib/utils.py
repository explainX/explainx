def is_classification(model):
        """ [Returns if the problem type is classification or regression]
        
        Args:
            model function
        
        Returns:
            True if classification
            False if regression
        """
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(model) & ~is_regressor(model):
            return True
        else:
            return False
import { useState } from 'react';
import { Star, Trash2 } from 'react-feather';
import './ItemIterator.css';

const SelectionDetails = ({
  id,
  key,
  item,
  handleSelectionClicked
}) => {
  const [isSelected, setIsSelected] = useState(false);
  const [isDeleteVisible, setIsDeleteVisible] = useState(false);

  const handleWhy = () => {
    setIsSelected(!isSelected);
  };

  const handleDelete = () => {
    setIsSelected(!isSelected);
  };

  const handleItemClick = () => {
    //handleSelectionClicked()
    setIsDeleteVisible(!isSelected);
  };

  return (
    <div className="item-selection" onClick={() => handleItemClick(item)}>
      <div className="item-set">
        <span className="author-selection">{item.author + ' /'}</span>
        <span className="repo-selection">{item.repository}</span>
      </div>
      <div className="item-set">
        {/* Add the star icon here */}
        <span className="stars-selection">
          <Star size={16} /> {item.stars}
        </span>
        <span className="update-selection">{item.update}</span>
      </div>
      {isDeleteVisible && (
        <div className="delete-overlay">
          <Trash2 size={24} onClick={handleDelete} />
        </div>
      )}

    </div>
  );
};

export default SelectionDetails;

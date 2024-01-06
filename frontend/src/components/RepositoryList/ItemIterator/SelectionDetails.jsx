import { useState, useEffect } from 'react';
import { Star, Trash2 } from 'react-feather';
import './ItemIterator.css';

/**
 * Component representing the details of a selection item.
 *
 * @component
 * @param {object} props - The component props.
 * @param {string} props.id - The unique identifier of the selection item.
 * @param {string} props.key - React key, reserved property in React components.
 * @param {object} props.item - The data object representing the selection item.
 * @param {function} props.handleSelectionClicked - Callback function when the item is clicked.
 * @param {string} props.selectedColor - The color to highlight the selected item.
 * @returns {JSX.Element} The rendered SelectionDetails component.
 */
const SelectionDetails = ({ id, key, item, handleSelectionClicked, selectedColor }) => {
  // State to manage the selection status of the item.
  const [isSelected, setIsSelected] = useState(false);

  // State to manage the visibility of the delete overlay.
  const [isDeleteVisible, setIsDeleteVisible] = useState(false);

  /**
   * Toggles the selection status of the item.
   */
  const handleWhy = () => {
    setIsSelected(!isSelected);
  };

  /**
   * Toggles the visibility of the delete overlay.
   */
  const handleDelete = () => {
    setIsSelected(!isSelected);
  };

  /**
   * Handles the click event on the selection item.
   *
   * @param {object} item - The data object representing the selection item.
   */
  const handleItemClick = (item) => {
    console.log(`LOG item: ${item}`);
    handleSelectionClicked(item);
    setIsDeleteVisible(!isSelected);
  };

  // Inline style to highlight the selected item with a colored shadow.
  const itemSelectionStyle = {
    boxShadow: `inset 8px 0 0 0 ${selectedColor}`,
  };

  // Rendering of the SelectionDetails component.
  return (
    <div className="item-selection" style={itemSelectionStyle} onClick={() => handleItemClick(item)}>
      <div className="item-set">
        <span className="author-selection">{item.author + ' /'}</span>
        <span className="repo-selection">{item.repository}</span>
      </div>
      <div className="item-set-details">
        {/* Add the star icon here */}
        <span className="stars-selection">
          <Star size={10.67} /> {item.stars}
        </span>
        <span className="update-selection">{item.update}</span>
      </div>
      {/* Render delete overlay if visible */}
      {isDeleteVisible && (
        <div className="delete-overlay">
          <Trash2 size={24} onClick={handleDelete} />
        </div>
      )}
    </div>
  );
};

export default SelectionDetails;

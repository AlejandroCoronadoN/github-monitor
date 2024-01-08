import React, { useState, useEffect } from 'react';
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
const SelectionDetails = ({
    id,
    key,
    item,
    handleSelectionClicked,
    selectedColor,
    setPlotsSeries,
    plotsSeries,
    setHoverIndex,
}) => {
  // State to manage the selection status of the item.
  const [isSelected, setIsSelected] = useState(false);

  // State to manage the visibility of the delete overlay.
  const [isDeleteVisible, setIsDeleteVisible] = useState(false);


  /**
   * Toggles the visibility of the delete overlay.
   */
  const handleDelete = () => {
    setIsSelected(!isSelected);

    // Find the index of the item to be deleted in plotSeries
    const indexToDelete = plotsSeries.findIndex((plotData) => plotData.id === item.id);

    if (indexToDelete !== -1) {
        // Remove the item from plotSeries
        setPlotsSeries((prevPlotSeries) => [
        ...prevPlotSeries.slice(0, indexToDelete),
        ...prevPlotSeries.slice(indexToDelete + 1),
        ]);

        // Update the ids to keep the enumeration
        setPlotsSeries((prevPlotSeries) =>
        prevPlotSeries.map((plotData, index) => ({
            ...plotData,
            id: index,
            }))
            );
        };
  }

  /**
   * Handles the click event on the selection item.
   *
   * @param {object} item - The data object representing the selection item.
   */
  const handleItemClick = (item) => {
    handleSelectionClicked(item);
    setIsDeleteVisible(!isSelected);
  };

  // Inline style to highlight the selected item with a colored shadow.
  const itemSelectionStyle = {
    boxShadow: `inset 8px 0 0 0 ${selectedColor}`,
  };

  // Rendering of the SelectionDetails component.
  return (
    <div className="item-selection"
        style={itemSelectionStyle}
        onClick={() => handleItemClick(item)}
        onMouseEnter={() => setHoverIndex(item.id)}
        onMouseLeave={() => setHoverIndex(item.id)}
        >
      <div className="item-set">
        <span className="author-selection">{item.author + ' /'}</span>
        <span className="repo-selection">{item.repository.slice(0,15)}</span>
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

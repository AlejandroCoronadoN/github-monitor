import React,  { useState } from "react";
import './ItemIterator.css';

/**
 * Component representing details of a suggestion item.
 *
 * @component
 * @param {object} props - The component props.
 * @param {number} props.id - The unique identifier for the suggestion item.
 * @param {string} props.item.author - The author of the repository.
 * @param {string} props.item.repository - The name of the repository.
 * @param {function} props.handleSuggestionClick - Function to handle the click event on the suggestion item.
 * @returns {JSX.Element} The rendered ItemDetails component.
 */
const ItemDetails = ({
    id,
    item,
    handleSuggestionClick
}) => {
    // State to manage the selection status of the suggestion item.
    const [isSelected, setIsSelected] = useState(false);

    return (
        <div  className="item-suggestion-click" onClick={() => handleSuggestionClick(item)}>
            <span className="author">{item.author + "/"}</span>
            <span className="repo">{item.repository}</span>
        </div>
    );
}

export default ItemDetails;
